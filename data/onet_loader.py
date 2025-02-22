import os
import pandas as pd
class ONETDataLoader:
    def __init__(self, data_dir="ONET"):
        self.data_dir = data_dir

    def load_datasets(self):
        datasets = {
            'occupations': pd.read_excel(f"{self.data_dir}/Occupation Data.xlsx"),
            'skills': pd.read_excel(f"{self.data_dir}/Skills.xlsx"),
            'interests': pd.read_excel(f"{self.data_dir}/Interests.xlsx"),
            'values': pd.read_excel(f"{self.data_dir}/Work Values.xlsx")
        }
        print("O*NET data loaded successfully!")
        merge = {

        }
        return datasets
    
    def load_merged_data(self):
        """Load preprocessed merged data if available."""
        if os.path.exists('data/merged_data.pkl'):
            return pd.read_pickle('data/merged_data.pkl')
        raise FileNotFoundError("Preprocessed data not found")

    def preprocess_datasets(self, datasets):
        """Preprocess O*NET datasets."""
        print("Preprocessing O*NET data...")
        processed = {}
        
        for key in ['skills', 'interests', 'values']:
            processed[key] = datasets[key].pivot_table(
                index='O*NET-SOC Code',
                columns='Element Name',
                values='Data Value',
                fill_value=0
            )

        occupation_basics = datasets['occupations'][['O*NET-SOC Code', 'Title', 'Description']]
        merged_data = occupation_basics.set_index('O*NET-SOC Code')

        for key in processed:
            if not processed[key].empty:
                merged_data = merged_data.join(processed[key], how='left')

        merged_data = merged_data.fillna(0)
        merged_data.to_pickle('data/merged_data.pkl')
        print("Data preprocessing complete!")
        return merged_data
    
ONETDataLoader()