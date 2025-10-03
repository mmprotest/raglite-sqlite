# How to tune hybrid alpha

- Run `raglite query --alpha 0.6` for balanced lexical/vector ranking.
- Increase to 0.8 for keyword-heavy compliance documents.
- Decrease to 0.3 when questions are phrased conversationally.
- Validate results against the built-in tiny eval dataset before rolling out.

Record the chosen alpha inside `raglite.toml` so the server and CLI stay aligned.
