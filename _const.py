dir_path = {
    "raw_data_dir" : "./",

    "select_relevant_features_dir" : 'csv_log/select_relevant_features',

    "handle_outliers_dir" : 'csv_log/handle_outliers',
    "detect_outliers_dir" : 'csv_log/handle_outliers' + '/detect',
    "remove_outliers_dir" : 'csv_log/handle_outliers' + '/remove',

    "handle_missing_value_dir" : 'csv_log/handle_missing_value',
    "remove_missing_value_dir" : 'csv_log/handle_missing_value' + '/remove',
    "fill_missing_value_dir" : 'csv_log/handle_missing_value' + '/fill', 

    "descriptive_statistics" : 'csv_log/descriptive_statistics',

    "final_training_data_dir" : 'csv_log/final_training_data',
    
    "final_test_data_dir" : 'csv_log/final_test_data',
    
    "predict_result_dir" : 'csv_log/predict_result'
}

train_features = [
                    'Basic_Demos-Age', 'Basic_Demos-Sex',
                    'CGAS-CGAS_Score', 
                    'Physical-BMI', 'Physical-Diastolic_BP', 
                    'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 
                    'FGC-FGC_PU', 'FGC-FGC_PU_Zone', 
                    'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 
                    'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone', 
                    'FGC-FGC_TL', 'FGC-FGC_TL_Zone',
                    'BIA-BIA_Activity_Level_num', 'BIA-BIA_DEE',
                    'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday',
                    'sii'
                ]

exclude_features_for_Outliers = [
                                    'Basic_Demos-Age', 'Basic_Demos-Sex', 
                                    'FGC-FGC_CU_Zone', 'FGC-FGC_PU_Zone', 
                                    'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 
                                    'FGC-FGC_TL_Zone', 'BIA-BIA_Activity_Level_num', 
                                    'PreInt_EduHx-computerinternet_hoursday',
                                    'sii'
                                ]
continuous_features = [
                        'Basic_Demos-Age',
                        'CGAS-CGAS_Score', 
                        'Physical-BMI', 'Physical-Diastolic_BP', 
                        'FGC-FGC_CU', 'FGC-FGC_PU',
                        'FGC-FGC_SRL', 'FGC-FGC_SRR', 
                        'FGC-FGC_TL', 'BIA-BIA_DEE', 
                        'SDS-SDS_Total_T',
                    ]

categorical_features = [
                            'Basic_Demos-Sex', 
                            'FGC-FGC_CU_Zone', 'FGC-FGC_PU_Zone', 
                            'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 
                            'FGC-FGC_TL_Zone', 'BIA-BIA_Activity_Level_num', 
                            'PreInt_EduHx-computerinternet_hoursday',
                        ]

input_features = [
                    'Basic_Demos-Age', 'Basic_Demos-Sex',
                    'CGAS-CGAS_Score', 
                    'Physical-BMI', 'Physical-Diastolic_BP', 
                    'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 
                    'FGC-FGC_PU', 'FGC-FGC_PU_Zone', 
                    'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 
                    'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone', 
                    'FGC-FGC_TL', 'FGC-FGC_TL_Zone',
                    'BIA-BIA_Activity_Level_num', 'BIA-BIA_DEE',
                    'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday',
                ]