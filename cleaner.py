# Load in packages
import numpy as np
import pandas as pd
from typing import List, Dict

import cleaner_utils

if __name__=="__main__":
    data_dir = "./data/"
    reference_path: str = data_dir + "Reference.csv"
    reference: pd.DataFrame = pd.read_csv(reference_path)

    cases: List[int] = [43, 44, 55, 56]

    # Aggregate ICP and grey value data for all sharks
    data_icp_df: pd.DataFrame = pd.DataFrame()
    data_grey_df: pd.DataFrame = pd.DataFrame()

    for case in cases:
        grey_values: pd.DataFrame = pd.read_csv(f"{data_dir}GreyValues_{case}.csv")
        icp_comp: pd.DataFrame = pd.read_csv(f"{data_dir}ICP_MS_Line_Data_{case}.csv")

        # Get birth pixel for alignment
        birth_pixel: int = reference.loc[reference['ALLOCATED ID'] == case, "Birth Mark Position (pixel)"].values[0]

        # Align ICP data to birth pixel
        icp_comp = icp_comp[icp_comp.columns[(icp_comp >= 0).all()]]
        icp_comp['pixel'] = icp_comp['Distance (um)'] / cleaner_utils.CONVERT_UM_TO_PIXEL
        data_icp: pd.DataFrame = icp_comp[icp_comp['pixel'] >= birth_pixel].reset_index(drop=True)
        data_icp['new_distance'] = data_icp['Distance (um)'] - data_icp['Distance (um)'].iloc[0]
        data_icp['new_pixel'] = data_icp['pixel'] - data_icp['pixel'].iloc[0]
        data_icp['case'] = case

        # Align grey value data to birth pixel
        data_grey: pd.DataFrame = grey_values[grey_values["Distance_(pixels)"] >= birth_pixel].reset_index(drop=True)
        data_grey['new_pixel'] = data_grey['Distance_(pixels)'] - data_grey['Distance_(pixels)'].iloc[0]
        data_grey['case'] = case

        data_icp_df = pd.concat([data_icp_df, data_icp], ignore_index=True)
        data_grey_df = pd.concat([data_grey_df, data_grey], ignore_index=True)

    # Remove columns with NaNs
    data_icp_df = data_icp_df.dropna(axis=1).reset_index(drop=True)
    data_grey_df = data_grey_df.dropna(axis=1).reset_index(drop=True)

    # Log-transform ICP data and remove outliers
    transformed_icp_data: pd.DataFrame = pd.DataFrame()

    for case in cases:
        temp_df: pd.DataFrame = pd.DataFrame()
        df: pd.DataFrame = data_icp_df[data_icp_df.case == case]
        for col in df.columns:
            if col in ['case', 'new_pixel', 'new_distance', 'pixel', 'Distance (um)']:
                temp_df[col] = df[col].values
                continue
            vals = cleaner_utils.average_around_0_for_log(df[col].values)
            cleaned = cleaner_utils.remove_outlier_with_emd(np.log(vals))
            temp_df[col] = cleaned if cleaned is not None else df[col].values
        temp_df['case'] = case
        transformed_icp_data = pd.concat([transformed_icp_data, temp_df], ignore_index=True)

        # Clean grey values
        df_grey: pd.DataFrame = data_grey_df[data_grey_df.case == case]
        cleaned_grey = cleaner_utils.remove_outlier_with_emd(np.log(df_grey['Gray_Value'].values))
        data_grey_df.loc[data_grey_df.case == case, 'cleaned_grey'] = cleaned_grey

    transformed_icp_data['new_pixel'] = data_icp_df['new_pixel'].values
    transformed_icp_data = transformed_icp_data.dropna(axis=1).reset_index(drop=True)

    # Downsample by averaging over pixel windows
    p: int = 5
    m: int = int(np.ceil(data_icp_df['new_pixel'].max()) + p)
    pixels: List[int] = list(range(p, m, p))

    ave_val_comp: pd.DataFrame = pd.DataFrame()
    grey_values: pd.DataFrame = pd.DataFrame()

    for case in cases:
        transformed_data: pd.DataFrame = transformed_icp_data[transformed_icp_data.case == case].reset_index(drop=True)
        averaged_data: pd.DataFrame = transformed_data # cleaner_utils.down_sampling_arrays(transformed_data, pixels, 'new_pixel')
        grey_case: pd.DataFrame = data_grey_df[data_grey_df.case == case].reset_index(drop=True)
        grey_case['case'] = case
        averaged_grey: pd.DataFrame = grey_case # cleaner_utils.down_sampling_arrays(grey_case, pixels, 'new_pixel')
        averaged_grey.subtract(averaged_grey.mean()).divide(averaged_grey.std())
        grey_values = pd.concat([grey_values, averaged_grey], ignore_index=True)
        averaged_data['case'] = case
        averaged_data.subtract(averaged_data.mean()).divide(averaged_data.std())
        ave_val_comp = pd.concat([ave_val_comp, averaged_data], ignore_index=True)

    # Combine ICP and grey data, standardize, and add metadata
    ave_val_comp = ave_val_comp.reset_index(drop=True)
    # ave_val_comp['grey_values'] = ave_grey
    # ave_val_comp['pixels_used'] = np.tile(pixels, len(cases))
    ave_val_comp = ave_val_comp.dropna()

    data_standard: pd.DataFrame = ave_val_comp.subtract(ave_val_comp.mean()).divide(ave_val_comp.std())
    data_standard['case'] = ave_val_comp['case']
    # data_standard['pixels_used'] = ave_val_comp['pixels_used']

    # Add shark sex
    sex_map: Dict[int, str] = {43: 'M', 44: 'M', 55: 'F', 56: 'F'}
    data_standard['sex'] = data_standard['case'].map(sex_map)

    # Save cleaned, standardized data
    data_standard.to_csv(f"{data_dir}cleaned_data.txt", index=False, header=True)
    
    grey_values['sex'] = grey_values['case'].map(sex_map)
    grey_values.to_csv(f"{data_dir}cleaned_grey_values.txt", index=False, header=True)
