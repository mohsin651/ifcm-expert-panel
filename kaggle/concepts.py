# Shared concept / diagnosis definitions (copied into each kaggle script at push time).

# key -> (dermoscopic description, short name used in negative prompts)
CONCEPTS = {
    "atypical_pn":       ("an atypical pigment network (irregular, thickened, broken lines)", "atypical pigment network"),
    "typical_pn":        ("a typical pigment network (regular, thin, evenly spaced lines)", "typical pigment network"),
    "irregular_streaks": ("irregular streaks (pseudopods or radial streaming unevenly distributed at the periphery)", "irregular streaks"),
    "regular_streaks":   ("regular streaks (symmetric starburst pattern of streaks around the whole periphery)", "regular streaks"),
    "irregular_pigm":    ("irregular blotches of pigmentation (asymmetric structureless dark areas)", "irregular pigmentation"),
    "regular_pigm":      ("regular pigmentation (homogeneous, symmetrically distributed structureless pigmentation)", "regular pigmentation"),
    "regression":        ("regression structures (white scar-like areas and/or blue-gray peppering)", "regression structures"),
    "irregular_dg":      ("irregular dots and globules (varying in size, shape and distribution)", "irregular dots and globules"),
    "regular_dg":        ("regular dots and globules (uniform in size and evenly distributed)", "regular dots and globules"),
    "bwv":               ("a blue-whitish veil (confluent blue pigmentation with an overlying white ground-glass haze)", "blue-whitish veil"),
    "arborizing":        ("arborizing (tree-like branching) vessels", "arborizing vessels"),
    "atypical_vessels":  ("atypical vascular structures (dotted or linear irregular vessels)", "atypical vessels"),
    "comma":             ("comma-shaped vessels", "comma vessels"),
    "hairpin":           ("hairpin vessels", "hairpin vessels"),
    "vessels_regr":      ("vessels within regression structures", "vessels within regression"),
}

DIAGNOSES = {
    "MEL": "melanoma",
    "NEV": "benign melanocytic nevus",
    "BCC": "basal cell carcinoma",
    "SK":  "seborrheic keratosis",
}
