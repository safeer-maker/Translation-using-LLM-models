import Model_Training
import Dataset


ds = Dataset.load_dataset_from_hf('kde4', lang1 = 'en', lang2 = 'fr' )

print (ds)