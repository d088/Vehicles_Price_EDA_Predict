def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Check if files are already extracted
if not os.path.exists("car_price_prediction_model.pkl"):
    extract_zip("car_price_prediction_model.zip", ".")
