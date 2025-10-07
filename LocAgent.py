from __future__ import annotations

import asyncio
import json
import tempfile
import typing as ty
from pathlib import Path

import torch
from Bio.Seq import Seq
from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.tools import ShellTool
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# 0.  LLM backbone – Llama-3-8B-Instruct quantized to 4-bit
# ------------------------------------------------------------------
LLAMA_BIN = Path("llama-3-8b-instruct-q4_0.gguf")   # download beforehand
assert LLAMA_BIN.exists(), "Place Llama-3-8B-Instruct q4_0.gguf in cwd"

llm = LlamaCpp(
    model_path=str(LLAMA_BIN),
    n_ctx=4096,
    temperature=0.15,
    max_tokens=1024,
    streaming=False,
    verbose=False,
)

# ------------------------------------------------------------------
# 1.  External tool wrappers
# ------------------------------------------------------------------
class ESMFoldStructureTool(BaseTool):
    name = "esmfold_structure_prediction"
    description = (
        "Predict the 3-D structure of a protein sequence with ESMFold-v1. "
        "Input: plain amino-acid string (FASTA). "
        "Output: dict with keys {pdb_path, plddt_list, sec_struct_list, rsa_list}."
    )

    async def _arun(
        self, sequence: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> dict:
        from esm import ESMFoldInferenceModule
        fold = ESMFoldInferenceModule(model_dir="esmfold_data")
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            pdb_path = tmp.name
        out = fold.infer(sequence, pdb_path)
        return {
            "pdb_path": pdb_path,
            "plddt_list": out["plddt"].cpu().tolist(),
            "sec_struct_list": out["sec_struct"],
            "rsa_list": out["rsa"].cpu().tolist(),
        }

    def _run(self, sequence: str) -> dict:
        return asyncio.run(self._arun(sequence))


class SLAttnESMLocTool(BaseTool):
    name = "slattnesm_localization"
    description = (
        "SL-AttnESM: a structure-aware multi-label classifier for 10 eukaryotic "
        "sub-cellular compartments. "
        "Input: dict returned by esmfold_structure_prediction. "
        "Output: dict with keys {comp_probs, attention_vector}."
    )

    async def _arun(self, payload: dict, **kwargs) -> dict:
        # pseudo-code – replace with real SL-AttnESM forward pass
        dummy_probs = {
            "Cytoplasm": 0.81,
            "Nucleus": 0.12,
            "Peroxisome": 0.05,
            "Mitochondrion": 0.02,
            "Endoplasmic reticulum": 0.15,
            "Cell membrane": 0.04,
            "Extracellular": 0.01,
            "Lysosome/Vacuole": 0.01,
            "Golgi apparatus": 0.01,
            "Plastid": 0.00,
        }
        dummy_attn = [0.0] * len(payload["plddt_list"])
        # highlight C-terminus as dummy example
        dummy_attn[-3:] = [0.9, 0.95, 1.0]
        return {"comp_probs": dummy_probs, "attention_vector": dummy_attn}

    def _run(self, payload: dict) -> dict:
        return asyncio.run(self._arun(payload))


class SortingSignalTool(BaseTool):
    name = "sorting_signal_scan"
    description = (
        "Scan sequence for classical sorting signals: SignalP-6 (signal peptide), "
        "NetGPI-3 (GPI-anchor), TMHMM-2 (trans-membrane helices). "
        "Input: amino-acid string. "
        "Output: unified JSON with keys {signal_peptide, gpi_anchor, transmembrane}."
    )

    async def _arun(self, sequence: str, **kwargs) -> dict:
        # dummy – replace with real binaries
        return {
            "signal_peptide": {"present": False, "cleavage": None},
            "gpi_anchor": {"present": False, "w_site": None},
            "transmembrane": {"helices": 0, "topology": None},
        }

    def _run(self, sequence: str) -> dict:
        return asyncio.run(self._arun(sequence))


class PrimerDesignTool(BaseTool):
    name = "primer_design_cloning"
    description = (
        "Design primer pairs for inserting a desired targeting motif at the native "
        "terminus while avoiding internal restriction sites. "
        "Input: dict with keys {sequence, motif, restriction_sites_to_avoid}. "
        "Output: list of ranked primer objects with keys {fwd, rev, tm, gc, cost}."
    )

    async def _arun(self, payload: dict, **kwargs) -> list[dict]:
        # dummy – replace with Primer-BLAST or primer3-py
        return [
            {
                "fwd": "GGAATTCCATG...",
                "rev": "CGGGATCCTTA...",
                "tm": 62.5,
                "gc": 0.52,
                "cost": 12.4,
            }
        ]

    def _run(self, payload: dict) -> list[dict]:
        return asyncio.run(self._arun(payload))


# ------------------------------------------------------------------
# 2.  Prompt templates – verbose English instructions
# ------------------------------------------------------------------
SYSTEM_PROMPT = """
You are 'LocAgent', an autonomous synthetic-biology assistant that turns
raw protein sequences into experiment-ready sub-cellular localization plans.
You ALWAYS follow the four-step pipeline below:

STEP-1  Structure prediction  
        → call `esmfold_structure_prediction` once.

STEP-2  Localization probabilities  
        → feed the structure dict to `slattnesm_localization`.

STEP-3  Sorting-signal scan  
        → run `sorting_signal_scan` on the original sequence.

STEP-4  Wet-lab plan  
        → if the user explicitly asks to *redirect* the protein to a
          specific compartment, decide which motif needs to be added
          (e.g. C-terminal SKL for peroxisomes, SV40-NLS for nucleus).
        → build the payload for `primer_design_cloning` with:
             - full amino-acid sequence  
             - desired motif  
             - restriction sites to avoid (EcoRI, BamHI by default)
        → return a ranked list of primer pairs, synthesis cost, and
          a short bench protocol (bullet points).

IMPORTANT:
- Never guess numerical values – always call the tools.
- Speak in concise, actionable English.
- After every tool call, briefly summarise what you learned.
- End your answer with a one-sentence takeaway for the biologist.
"""

HUMAN_TEMPLATE = """
Protein sequence:
{sequence}

User request:
{user_request}

Begin the LocAgent pipeline now.
"""

prompt = PromptTemplate(
    input_variables=["sequence", "user_request"],
    template=HUMAN_TEMPLATE,
)

# ------------------------------------------------------------------
# 3.  Bind tools into LangChain agent
# ------------------------------------------------------------------
toolbox = [
    ESMFoldStructureTool(),
    SLAttnESMLocTool(),
    SortingSignalTool(),
    PrimerDesignTool(),
]

agent_chain = initialize_agent(
    tools=toolbox,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=8,
    early_stopping_method="generate",
    agent_kwargs={"prefix": SYSTEM_PROMPT},
)

# ------------------------------------------------------------------
# 4.  Convenience wrapper
# ------------------------------------------------------------------
async def run_locagent(sequence: str, user_request: str = "Predict localization.") -> str:
    """Async entry point for the LocAgent pipeline."""
    return await agent_chain.arun(
        prompt.format(sequence=sequence, user_request=user_request)
    )