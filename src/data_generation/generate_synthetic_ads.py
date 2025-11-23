import numpy as np
import pandas as pd

def generate_synthetic_ads(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    ages = np.random.randint(18, 60, n_samples)
    genders = np.random.choice(["M", "F"], n_samples)
    city_tier = np.random.choice([1, 2, 3], n_samples)

    colourfulness = np.random.uniform(1, 5, n_samples)
    emotional_appeal = np.random.uniform(1, 5, n_samples)
    message_clarity = np.random.uniform(1, 5, n_samples)
    duration = np.random.randint(10, 60, n_samples)

    ad_recall = np.random.randint(10, 100, n_samples)
    likeability = np.random.uniform(1, 5, n_samples)
    purchase_intent = np.random.uniform(1, 5, n_samples)

    ctr = np.random.uniform(0.001, 0.05, n_samples)
    vcr = np.random.uniform(0.2, 0.95, n_samples)

    performance = []
    for i in range(n_samples):
        score = (
            0.2 * ad_recall[i] +
            5 * likeability[i] +
            5 * purchase_intent[i] +
            100 * ctr[i] +
            50 * vcr[i]
        )
        if score > 120:
            performance.append("High")
        elif score > 70:
            performance.append("Medium")
        else:
            performance.append("Low")

    df = pd.DataFrame({
        "Age": ages,
        "Gender": genders,
        "City_Tier": city_tier,
        "Colourfulness": colourfulness,
        "Emotional_Appeal": emotional_appeal,
        "Message_Clarity": message_clarity,
        "Duration": duration,
        "Ad_Recall": ad_recall,
        "Likeability": likeability,
        "Purchase_Intent": purchase_intent,
        "CTR": ctr,
        "VCR": vcr,
        "Ad_Performance": performance
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_ads()
    df.to_csv("data/raw/synthetic_ads_raw.csv", index=False)
    print("Synthetic dataset saved to data/raw/")
