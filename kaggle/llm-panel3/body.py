# LLM panel, bias-reduced question styles:
#  "comp":   per (finding, diagnosis): is the diagnosis MORE / LESS / EQUALLY common among lesions with the finding than without it?
#  "choice": per finding: which of the 4 diagnoses is most likely given the finding (forces relative allocation)?
# Votes are read from next-token probabilities of the option letters (no sampling).
import os, gc, shutil, time, itertools, json
import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

SMOKE = os.environ.get("SMOKE", "0") == "1"
PART = int(os.environ.get("PART", "0"))
MODEL_SETS = [
    ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "allenai/OLMo-2-1124-7B-Instruct",
     "microsoft/Phi-3.5-mini-instruct", "aaditya/Llama3-OpenBioLLM-8B"],
    ["m42-health/Llama3-Med42-8B", "01-ai/Yi-1.5-9B-Chat", "ibm-granite/granite-3.1-8b-instruct",
     "tiiuae/Falcon3-7B-Instruct", "microsoft/Phi-4-mini-instruct"],
]
MODELS = MODEL_SETS[PART]
if os.environ.get("IFCM_MODELS"):
    selected = [name.strip() for name in os.environ["IFCM_MODELS"].split(",") if name.strip()]
    if not selected or any(name not in MODELS for name in selected):
        raise ValueError("IFCM_MODELS must name one or more models from this job's part")
    MODELS = selected
OUT = os.environ.get("IFCM_OUT", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)
LOCAL_MODELS = json.load(open(os.environ["IFCM_MODEL_MANIFEST"])) if os.environ.get("IFCM_MODEL_MANIFEST") else {}
LLAMA3_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}"
                   "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")

COMP_OPTIONS = {"up": "More common among lesions with this finding",
                "down": "Less common among lesions with this finding",
                "none": "About equally common with or without this finding",
                "unsure": "Uncertain / not established"}
COMP_ORDERS = [["up", "down", "none", "unsure"], ["unsure", "none", "down", "up"], ["none", "up", "unsure", "down"]]
COMP_TEMPLATES = [
    "You are an expert dermatologist. Compare skin lesions that show {c} on dermoscopy with lesions that do not show it. Is {d} more common, less common, or about equally common among the lesions that show this finding?",
    "In a dermoscopy clinic, consider all lesions with {c} versus all lesions without it. How does the frequency of {d} differ between the two groups?",
    "Dermoscopy exam question: relative to lesions lacking {c}, is {d} over-represented, under-represented, or neither among lesions that have it?",
]
CHOICE_TEMPLATES = [
    "You are an expert dermatologist. A skin lesion shows {c} on dermoscopy. Which diagnosis is this finding most characteristic of?",
    "Dermoscopy exam question: {c} is most typically seen in which of the following lesions?",
    "If the only dermoscopic finding you know is {c}, which diagnosis is most likely?",
]
DKEYS = list(DIAGNOSES)
CHOICE_ORDERS = [DKEYS + ["unsure"], ["unsure"] + DKEYS[::-1], [DKEYS[2], "unsure", DKEYS[0], DKEYS[3], DKEYS[1]]]

concepts = list(CONCEPTS.items())
diags = list(DIAGNOSES.items())
if SMOKE:
    MODELS, concepts, diags = MODELS[:1], concepts[:2], diags[:2]

def prompt(tok, q, labels):
    letters = "ABCDE"[:len(labels)]
    q += "\n\n" + "\n".join(f"{letters[i]}) {t}" for i, t in enumerate(labels))
    q += f"\n\nAnswer with a single letter ({', '.join(letters)})."
    if tok.chat_template is None:
        tok.chat_template = LLAMA3_TEMPLATE
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "Answer:"

def letter_probs(model, tok, text, n):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits[0, -1].float(), -1)
    mass = []
    for L in "ABCDE"[:n]:
        ids = {tok.encode(L, add_special_tokens=False)[0], tok.encode(" " + L, add_special_tokens=False)[-1]}
        mass.append(probs[sorted(ids)].sum().item())
    tot = sum(mass)
    return [m / tot for m in mass], tot

comp_rows, choice_rows, failed_models = [], [], []
for mname in MODELS:
    t0 = time.time()
    try:
        source = LOCAL_MODELS[mname]["snapshot"] if LOCAL_MODELS else mname
        tok = AutoTokenizer.from_pretrained(source)
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float16, device_map="auto").eval()
        for (ck, (cdesc, _)), (dk, ddesc), t, oi in itertools.product(concepts, diags, range(3), range(3)):
            order = COMP_ORDERS[oi]
            p, tot = letter_probs(model, tok, prompt(tok, COMP_TEMPLATES[t].format(c=cdesc, d=ddesc), [COMP_OPTIONS[o] for o in order]), 4)
            r = {"model": mname, "concept": ck, "diag": dk, "template": t, "order": oi, "letter_mass": tot}
            r.update({"p_" + o: p[i] for i, o in enumerate(order)})
            comp_rows.append(r)
        for (ck, (cdesc, _)), t, oi in itertools.product(concepts, range(3), range(3)):
            order = CHOICE_ORDERS[oi]
            labels = [DIAGNOSES[o] if o != "unsure" else "Cannot tell / not characteristic of any of these" for o in order]
            p, tot = letter_probs(model, tok, prompt(tok, CHOICE_TEMPLATES[t].format(c=cdesc), labels), 5)
            r = {"model": mname, "concept": ck, "template": t, "order": oi, "letter_mass": tot}
            r.update({"p_" + o: p[i] for i, o in enumerate(order)})
            choice_rows.append(r)
        cm = pd.DataFrame(comp_rows).query("model==@mname").letter_mass.mean()
        hm = pd.DataFrame(choice_rows).query("model==@mname").letter_mass.mean()
        print(f"{mname}: letter mass comp {cm:.3f} choice {hm:.3f}, {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"FAILED {mname}: {repr(e)}", flush=True)
        failed_models.append(mname)
    finally:
        for v in ["model", "tok"]:
            if v in globals(): del globals()[v]
        gc.collect(); torch.cuda.empty_cache()
    pd.DataFrame(comp_rows).to_csv(f"{OUT}/llm_comp_{PART}.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(f"{OUT}/llm_choice_{PART}.csv", index=False)
if failed_models:
    raise RuntimeError(f"Incomplete LLM panel: {failed_models}")
