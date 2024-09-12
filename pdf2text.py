import pymupdf4llm
import pathlib

md_text = pymupdf4llm.to_markdown("C:/Users/Admin/Desktop/00.-Employee-Handbook-FPT-Digital.pdf")

pathlib.Path("output.md").write_bytes(md_text.encode())
