# Source Reconstruction Notes

## Basis

The Python files in this package were reconstructed from the uploaded 530-page PDF titled
`ACES-FL IMPLEMENTATION SOURCE CODE`.

The PDF explicitly labels source-file boundaries with paths such as:

- `code/acesfl-baseline/...`
- `code/acesfl-selection/...`
- `code/acesfl-selection-v2/...`
- `code/acesfl-compression/...`
- `code/acesfl-security/...`
- `code/acesfl-defense/...`
- `code/generate_thesis_results.py`

## Repairs made

PDF text preserves the source indentation well, but visual page wrapping introduced a small
number of extraction artifacts. The reconstruction process:

1. preserved indentation and source text from the PDF;
2. removed PDF separator lines used only to label files;
3. repaired visual wrapping of long path strings in `generate_thesis_results.py`;
4. excluded `quickstart-pytorch` from the curated GitHub package because it is a Flower
   quickstart/template folder rather than an ACES-FL thesis experiment;
5. excluded superseded `*_before_*` backup variants from the curated package;
6. retained those files in the separate full archival reconstruction;
7. syntax-compiled every reconstructed Python file.

No missing `pyproject.toml`, dataset files, result CSVs, or run configuration files were fabricated.
