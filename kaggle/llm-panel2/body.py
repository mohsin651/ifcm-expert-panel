# LLM "expert panel": each open LLM votes on how each dermoscopic finding affects each diagnosis.
# Votes are read from next-token probabilities of the option letters (no sampling).
import os, gc, json, shutil, time, itertools
import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

SMOKE = os.environ.get("SMOKE", "0") == "1"
MODELS = [
    "aaditya/Llama3-OpenBioLLM-8B",
    "m42-health/Llama3-Med42-8B",
    "Intelligent-Internet/II-Medical-8B",
    "01-ai/Yi-1.5-9B-Chat",
    "ibm-granite/granite-3.1-8b-instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen3-8B",
]
if os.environ.get("IFCM_MODELS"):
    selected = [name.strip() for name in os.environ["IFCM_MODELS"].split(",") if name.strip()]
    if not selected or any(name not in MODELS for name in selected):
        raise ValueError("IFCM_MODELS must name one or more models from this job")
    MODELS = selected
OUT = os.environ.get("IFCM_OUT", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)
LOCAL_MODELS = json.load(open(os.environ["IFCM_MODEL_MANIFEST"])) if os.environ.get("IFCM_MODEL_MANIFEST") else {}

# option label -> meaning; order is shuffled per prompt to cancel letter bias
OPTIONS = {"up": "It increases the likelihood",
           "down": "It decreases the likelihood",
           "none": "It has no meaningful effect on the likelihood",
           "unsure": "This is uncertain / not established in the literature"}
ORDERS = [["up", "down", "none", "unsure"], ["unsure", "none", "down", "up"], ["none", "up", "unsure", "down"]]
TEMPLATES = [
    "You are an expert dermatologist. In dermoscopy, a skin lesion shows {c}. How does the presence of this finding change the likelihood that the lesion is a {d}?",
    "Consider dermoscopic diagnosis of pigmented and non-pigmented skin lesions. If {c} is observed, what is its effect on the probability of {d}?",
    "Question from a dermoscopy exam: what is the diagnostic significance of {c} for {d}?",
]
LETTERS = "ABCD"
LLAMA3_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}"
                   "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")

concepts = list(CONCEPTS.items())
diags = list(DIAGNOSES.items())
if SMOKE:
    MODELS, concepts, diags = MODELS[:1], concepts[:2], diags[:2]

def build(tok, c, d, t, order):
    q = TEMPLATES[t].format(c=c, d=d) + "\n\n"
    q += "\n".join(f"{LETTERS[i]}) {OPTIONS[o]}" for i, o in enumerate(order))
    q += "\n\nAnswer with a single letter (A, B, C or D)."
    msgs = [{"role": "user", "content": q}]
    if tok.chat_template is None:   # e.g. OpenBioLLM ships without one; it is Llama-3 based
        tok.chat_template = LLAMA3_TEMPLATE
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return p + "Answer:"

def letter_ids(tok):
    ids = []
    for L in LETTERS:
        cand = {tok.encode(L, add_special_tokens=False)[0], tok.encode(" " + L, add_special_tokens=False)[-1]}
        ids.append(sorted(cand))
    return ids

rows, failed_models = [], []
for mname in MODELS:
    t0 = time.time()
    try:
        source = LOCAL_MODELS[mname]["snapshot"] if LOCAL_MODELS else mname
        tok = AutoTokenizer.from_pretrained(source)
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float16, device_map="auto")
        model.eval()
        lids = letter_ids(tok)
        jobs = list(itertools.product(range(len(concepts)), range(len(diags)), range(len(TEMPLATES)), range(len(ORDERS))))
        for ci, di, t, oi in jobs:
            ck, (cdesc, _) = concepts[ci]
            dk, ddesc = diags[di]
            order = ORDERS[oi]
            enc = tok(build(tok, cdesc, ddesc, t, order), return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[0, -1].float()
            probs = torch.softmax(logits, -1)
            mass = [probs[ids].sum().item() for ids in lids]
            tot = sum(mass)
            r = {"model": mname, "concept": ck, "diag": dk, "template": t, "order": oi, "letter_mass": tot}
            for i, o in enumerate(order):
                r["p_" + o] = mass[i] / tot
            rows.append(r)
        print(f"{mname}: {len(jobs)} prompts, mean letter mass {pd.DataFrame(rows).query('model==@mname').letter_mass.mean():.3f}, {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"FAILED {mname}: {repr(e)}", flush=True)
        failed_models.append(mname)
    finally:
        for v in ["model", "tok"]:
            if v in globals(): del globals()[v]
        gc.collect(); torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(f"{OUT}/llm_votes2.csv", index=False)

if failed_models:
    raise RuntimeError(f"Incomplete LLM panel: {failed_models}")
df = pd.DataFrame(rows)
print(df.groupby(["model", "diag"])[["p_up", "p_down", "p_none", "p_unsure"]].mean().round(2).to_string())
