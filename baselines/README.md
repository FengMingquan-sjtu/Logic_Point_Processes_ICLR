# data
## crime 

path :``Learn_Logic_PP/Shuang_sythetic data codes/data/crime_all_week.npy``

num samples: 1860 (1500 training + 360 testing)

num preds: 14

pred names: ["SPRING", "SUMMER", "AUTUMN", "WINTER", "WEEKDAY", "WEEKEND", "MORNING", "AFTERNOON", "EVENING", "NIGHT", "VANDALISM", "LARCENY THEFT FROM MV - NON-ACCESSORY", "ASSAULT - SIMPLE", "LARCENY SHOPLIFTING"
]

target pred: "LARCENY SHOPLIFTING" (idx=13)

## mimic
path ``Learn_Logic_PP/Shuang_sythetic data codes/data/mimic_1.npy``

num samples: 2000 (1600 training + 400 testing)

num preds: 62

pred names: ['sysbp_low', 'spo2_sao2_low', 'cvp_low', 'svr_low', 'potassium_meql_low', 'sodium_low', 'chloride_low', 'bun_low', 'creatinine_low', 'crp_low', 'rbc_count_low', 'wbc_count_low', 'arterial_ph_low', 'arterial_be_low', 'arterial_lactate_low', 'hco3_low', 'svo2_scvo2_low', 'sysbp_normal', 'spo2_sao2_normal', 'cvp_normal', 'svr_normal', 'potassium_meql_normal', 'sodium_normal', 'chloride_normal', 'bun_normal', 'creatinine_normal', 'crp_normal', 'rbc_count_normal', 'wbc_count_normal', 'arterial_ph_normal', 'arterial_be_normal', 'arterial_lactate_normal', 'hco3_normal', 'svo2_scvo2_normal', 'sysbp_high', 'spo2_sao2_high', 'cvp_high', 'svr_high', 'potassium_meql_high', 'sodium_high', 'chloride_high', 'bun_high', 'creatinine_high', 'crp_high', 'rbc_count_high', 'wbc_count_high', 'arterial_ph_high', 'arterial_be_high', 'arterial_lactate_high', 'hco3_high', 'svo2_scvo2_high', 'real_time_urine_output_low', 'or_colloid', 'or_crystalloid', 'oral_water', 'norepinephrine_norad_levophed', 'epinephrine_adrenaline', 'dobutamine', 'dopamine', 'phenylephrine_neosynephrine', 'milrinone', 'survival'
]

target pred: 'real_time_urine_output_low' (idx=51)


# data preprocessing
Our data are stored as nested dict: data[sample_ID][pred_ID]={"time":time_list, "state":state_list}. In baseline experiments, we need to convert data into required formats.

# baseline
The folder ``Learn_Logic_PP/baselines`` contains 3 sub-folders, corresponds to following 3 baselines.
## cause
data convertion is done.

totaling 4 models: HExp, HSG, RPPN, ERPP.

training command is stored in ``Learn_Logic_PP/baselines/cause/cmd.txt``

## tree
[todo] data conversion. 

the folder may contain files named ``mimic*``, but they are used for old mimic, we should write new verison.

training command is stored in ``Learn_Logic_PP/baselines/tree/cmd.txt``

## bn
[todo] data conversion. 

the folder may contain files named ``mimic*``, but they are used for old mimic, we should write new verison.

training command is stored in ``Learn_Logic_PP/baselines/tree/cmd.txt``



