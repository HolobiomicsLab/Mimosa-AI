"""Post-run export of the best workflow's agent trace into ASTRA YAML.

See https://astra-spec.org/latest/ for the schema. The exporter walks the
memory artefacts of the best evolved run, surfaces methodological decisions
via an LLM pass, and writes ``astra.yaml`` + ``universes/best.yaml`` into
the restored workspace alongside the analysis output.
"""

from sources.transparency.astra_exporter import AstraExporter

__all__ = ["AstraExporter"]
