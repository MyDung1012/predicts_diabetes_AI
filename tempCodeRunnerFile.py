import pickle

with open("GAN+LSTM.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))
print(obj)
