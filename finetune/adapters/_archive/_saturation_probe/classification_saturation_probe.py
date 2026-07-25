"""Draft: harder 10-class document generators with realistic confusable clusters.
Financial cluster (invoice/receipt/purchase_order/quote) shares one structural
template varying only a couple of phrase tokens -> genuine ambiguity.
Correspondence (email/memo/letter): email & memo both carry TO/FROM.
Legal (contract/nda): both are clause agreements; contract also has a
confidentiality clause overlapping nda.
"""
import random

_FIRST = ["Ava","Liam","Noah","Mia","Priya","Kenji","Sofia","Omar","Elena","Marcus","Aisha","Diego","Hana","Tariq","Lena","Ravi"]
_LAST = ["Osei","Nakamura","Delgado","Kowalski","Abebe","Rossi","Haddad","Fischer","Okonkwo","Petrov","Silva","Ahmed","Novak","Reyes"]
_COMPANY = ["Northwind Traders","Acme Logistics","Blue Harbor Foods","Meridian Analytics","Cedar & Vale LLP","Orbital Systems","Sunfield Grocers","Ironbridge Consulting","Halcyon Media"]
_CITY = ["Portland","Leeds","Nagoya","Turin","Accra","Valencia","Austin","Kraków","Lyon","Pune"]
_ITEM = ["Wireless mouse","USB-C hub","Standing desk","Notebook (pack of 5)","Ergonomic chair","Monitor arm","Coffee beans 1kg","Whiteboard","HDMI cable","Desk lamp"]
_ROLE = ["Data Engineer","Financial Analyst","Registered Nurse","Product Manager","Backend Developer","Operations Lead"]

def _money(r): return f"${r.randint(5,4999)}.{r.randint(0,99):02d}"
def _date(r): return f"2026-{r.randint(1,12):02d}-{r.randint(1,28):02d}"
def _num(r): return r.randint(1000,99999)
def _party(r): return f"{r.choice(_FIRST)} {r.choice(_LAST)}"

def _fin_body(r):
    """Shared financial skeleton: number, date, two parties, line items, total."""
    a, b = r.choice(_COMPANY), _party(r)
    lines = [f"  {r.choice(_ITEM)}  x{r.randint(1,6)}   {_money(r)}" for _ in range(r.randint(2,4))]
    return a, b, lines

def gen_invoice(r):
    a,b,lines=_fin_body(r)
    head = r.choice([f"INVOICE #{_num(r)}", f"#{_num(r)}", "Statement of charges"])
    return (f"{head}\nFrom: {a}\nBill to: {b}\nDate: {_date(r)}\n"+"\n".join(lines)+
            f"\nSubtotal {_money(r)}   Tax {_money(r)}\n"
            f"{r.choice(['Amount due','Balance due','Please remit'])}: {_money(r)}\n"
            f"{r.choice(['Net 30.','Payment due within 30 days.','Terms: Net 15.'])}")

def gen_receipt(r):
    a,b,lines=_fin_body(r)
    head = r.choice([f"{a}", f"{a} — Store #{r.randint(1,80)}"])
    return (f"{head}\n{_date(r)} {r.randint(8,21)}:{r.randint(0,59):02d}\n"+"\n".join(lines)+
            f"\nTotal {_money(r)}   {r.choice(['Cash','VISA ****'+str(r.randint(1000,9999)),'Card'])}\n"
            f"{r.choice(['Change due','Tendered'])} {_money(r)}\n"
            f"{r.choice(['Thank you for your purchase!','Thanks for shopping with us!','Paid — thank you.'])}")

def gen_purchase_order(r):
    a,b,lines=_fin_body(r)
    head = r.choice([f"PURCHASE ORDER #{_num(r)}", f"PO #{_num(r)}", f"Order #{_num(r)}"])
    return (f"{head}\nVendor: {a}\nShip to: {b}\nPO date: {_date(r)}\n"
            f"{r.choice(['Please supply the following:','Kindly deliver the items below:'])}\n"+"\n".join(lines)+
            f"\nOrder total {_money(r)}\nRequested delivery: {_date(r)}")

def gen_quote(r):
    a,b,lines=_fin_body(r)
    head = r.choice([f"QUOTATION #{_num(r)}", f"Quote #{_num(r)}", f"Estimate #{_num(r)}"])
    return (f"{head}\nPrepared by: {a}\nFor: {b}\nDate: {_date(r)}\n"+"\n".join(lines)+
            f"\nEstimated total {_money(r)}\n"
            f"{r.choice(['Valid until','Quote expires'])}: {_date(r)}\n"
            f"{r.choice(['This is not an invoice.','Prices subject to change.'])}")

def gen_email(r):
    a,b=_party(r),_party(r)
    return (f"From: {a.lower().replace(' ','.')}@{r.choice(['mail','corp','inbox'])}.com\n"
            f"To: {b.lower().replace(' ','.')}@example.com\n"
            f"Subject: Re: {r.choice(['Q3 numbers','the meeting','project update','your request'])}\n"
            f"Date: {_date(r)}\n\nHi {b.split()[0]},\nThanks for the note. I'll send the "
            "files by end of week. Let me know if anything changes.\n"
            f"Best,\n{a.split()[0]}")

def gen_memo(r):
    return (f"MEMORANDUM\nTO: {_party(r)}\nFROM: {_party(r)}\nDATE: {_date(r)}\n"
            f"RE: {r.choice(['Policy update','Q3 planning','Office closure','New procedure'])}\n\n"
            "Please be advised of the following changes effective next week. "
            "Direct any questions to your team lead.")

def gen_letter(r):
    a,comp=_party(r),r.choice(_COMPANY)
    return (f"{comp}\n{r.randint(1,999)} High Street, {r.choice(_CITY)}\n{_date(r)}\n\n"
            f"Dear {a},\nWe are writing to inform you of an update to your account. "
            "Please review the enclosed information at your earliest convenience.\n"
            f"Yours sincerely,\n{_party(r)}\nCustomer Relations, {comp}")

def gen_contract(r):
    a,b=r.choice(_COMPANY),r.choice(_COMPANY)
    return (f"{r.choice(['SERVICE AGREEMENT','AGREEMENT','CONSULTING AGREEMENT'])}\n"
            f'This Agreement is entered into as of {_date(r)} by and between {a} '
            f'("Provider") and {b} ("Client").\n'
            f"1. TERM. {r.randint(6,36)} months.\n2. FEES. {_money(r)} per month.\n"
            "3. CONFIDENTIALITY. Each party shall keep confidential information secret.\n"
            "4. TERMINATION. Either party may terminate with 30 days notice.\n"
            "IN WITNESS WHEREOF, the parties have executed this Agreement.")

def gen_nda(r):
    a,b=r.choice(_COMPANY),r.choice(_COMPANY)
    return (f"{r.choice(['NON-DISCLOSURE AGREEMENT','MUTUAL NDA','CONFIDENTIALITY AGREEMENT'])}\n"
            f'This Agreement is entered into as of {_date(r)} between {a} '
            f'("Disclosing Party") and {b} ("Receiving Party").\n'
            "1. The Receiving Party shall not disclose the Confidential Information.\n"
            f"2. TERM. Obligations survive for {r.randint(1,5)} years.\n"
            "3. Confidential Information excludes publicly available information.\n"
            "IN WITNESS WHEREOF, the parties have executed this Agreement.")

GENERATORS = {
    "invoice":gen_invoice,"receipt":gen_receipt,"purchase_order":gen_purchase_order,
    "quote":gen_quote,"email":gen_email,"memo":gen_memo,"letter":gen_letter,
    "contract":gen_contract,"nda":gen_nda,"resume":lambda r:(
        f"{_party(r)}\n{r.choice(_ROLE)} | {r.choice(_CITY)}\nSUMMARY\n"
        f"{r.randint(2,12)} years of experience.\nEXPERIENCE\n{r.choice(_ROLE)}, "
        f"{r.choice(_COMPANY)}\nEDUCATION\nB.Sc.\nSKILLS\nPython, SQL, leadership."),
}
LABELS = list(GENERATORS)

if __name__=="__main__":
    # Cheap separability proxy: TF-IDF + logistic regression (NOT the LoRA model,
    # just a sanity gauge that the task is non-trivial / below 1.00).
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    r=random.Random(0)
    X=[];y=[]
    for lab in LABELS:
        for _ in range(80):
            X.append(GENERATORS[lab](r)); y.append(lab)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=1,stratify=y)
    v=TfidfVectorizer(ngram_range=(1,2),min_df=2)
    clf=LogisticRegression(max_iter=1000)
    clf.fit(v.fit_transform(Xtr),ytr)
    pred=clf.predict(v.transform(Xte))
    print(f"TF-IDF+LR proxy  acc={accuracy_score(yte,pred):.3f}  macroF1={f1_score(yte,pred,average='macro'):.3f}")
    # per-class errors to see which clusters confuse
    from collections import Counter
    err=Counter()
    for t,p in zip(yte,pred):
        if t!=p: err[(t,p)]+=1
    print("top confusions:", err.most_common(8))
