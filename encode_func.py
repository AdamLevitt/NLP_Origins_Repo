from sklearn.preprocessing import LabelEncoder

def int_encode(enco):
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(enco)
  return integer_encoded