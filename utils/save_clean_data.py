from utils.clean_dataset import load_and_clean

def save_clean(input_path, output_path):
    df = load_and_clean(input_path)
    df.to_csv(output_path, index=False)
    print("Saved cleaned dataset!")

if __name__ == "__main__":
    save_clean("data/spam_assassin.csv", "data/spam_clean.csv")