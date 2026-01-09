# def create_target_column(df):
#     disease_cols = [c for c in df.columns if c.startswith('Disease_')]
    
#     def extract_disease(row):
#         for col in disease_cols:
#             if row[col] == 1:
#                 return col.replace('Disease_', '')
#         return 'No Disease'
    
#     df['Target'] = df.apply(extract_disease, axis=1)
#     df.drop(columns=disease_cols, inplace=True)
#     return df

# src/target_builder.py

# def create_target_column(df):
#     disease_cols = [c for c in df.columns if c.startswith("Disease_")]

#     def extract_disease(row):
#         for col in disease_cols:
#             if row[col] == 1:
#                 return col.replace("Disease_", "")
#         return "Unknown"

#     # Create target column
#     df["target_disease"] = df.apply(extract_disease, axis=1)

#     # Drop original disease columns
#     df = df.drop(columns=disease_cols)

#     # ✅ REMOVE 'Unknown' TARGETS (IMPORTANT)
#     df = df[df["target_disease"] != "Unknown"].reset_index(drop=True)

#     return df


from src.category_mapper import DISEASE_CATEGORY_MAP, DEFAULT_CATEGORY

def create_targets(df):
    disease_cols = [c for c in df.columns if c.startswith("Disease_")]

    def extract_disease(row):
        for col in disease_cols:
            if row[col] == 1:
                return col.replace("Disease_", "")
        return "Unknown"

    df["target_disease"] = df.apply(extract_disease, axis=1)

    df["target_category"] = df["target_disease"].apply(
        lambda x: DISEASE_CATEGORY_MAP.get(x, DEFAULT_CATEGORY)
    )

    df = df.drop(columns=disease_cols)

    return df
