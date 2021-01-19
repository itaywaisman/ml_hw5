
def save_selected_columns(original_list, final_list):
    df = pd.DataFrame([1 if feature in final_list else 0 for feature in original_list], index=original_list)
    
    df.T.to_csv('selected_columns.csv')