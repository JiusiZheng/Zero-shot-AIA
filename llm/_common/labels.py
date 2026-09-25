"""Attribute label sets shared by every LLM-supported attack script (dev/test,
original/optimized-prompt)."""

PARSE_ERROR_LABEL = "PARSE_ERROR"  # sentinel: never equals a true label -> always scored wrong

ATTRIBUTE_VALID_LABELS = {
    "gender": ["male_masculine", "female_feminine"],
    "age": ["less than 31", "31 to 50", "more than 50"],
    "accent": ["India and South Asia", "England", "Australian", "Canadian", "United States"],
}

# LaTeX-table recall-column order; age/accent order differs from ATTRIBUTE_VALID_LABELS above.
TABLE_COLUMNS = {
    "gender": [("male_masculine", "Male"), ("female_feminine", "Female")],
    "age": [("more than 50", "$>$50"), ("less than 31", "$<$31"), ("31 to 50", "31--50")],
    "accent": [
        ("India and South Asia", "Ind./S. Asia"),
        ("England", "England"),
        ("Australian", "Aus."),
        ("Canadian", "Can."),
        ("United States", "US"),
    ],
}
