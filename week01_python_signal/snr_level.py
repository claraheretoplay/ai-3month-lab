def classify_snr(snr):
    if snr < 0:
        return "low"
    elif snr < 15:
        return "medium"
    else:
        return "high"

for snr in [-5, 0, 10, 20]:
    print(snr, classify_snr(snr))