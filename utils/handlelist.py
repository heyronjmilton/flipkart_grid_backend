import time, os, json
from datetime import datetime, date, timedelta
import pytz

timezone = pytz.timezone('Asia/Kolkata')

class_names_list = ['act2_chillisurprise_40g', 'act2_goldensizzler_60g', 'aer_spraymusk_220ml', 'ajantas_ORS_21', 'ashirvaad_salt_500g', 'boost_200g', 'britannia_gobbles_50g', 'brut_deo_200ml', 'cammery_icecreamcoconut_1l', 'candid_dustingpowder_120g', 'chings_noodlesschezwan_240g', 'chocos_22g', 'cremebake_chocobrownie_18g', 'dhankary_nutsanddryfruits', 'eggs_pack0f6', 'finemustard_100g', 'gatsby_superhard5_30g', 'haldi_50g', 'himalaya_neemfacewash_100ml', 'inchi_tomatosoup_15g', 'kinder_creamy_19g', 'kismis_dmart', 'kitchenteasure_chillipowder_100g', 'kurkure_masalamunch_40g', 'lakerol_car_shampoo_200ml', 'lays_blue_24g', 'lays_green_24g', 'lays_tomato_24g', 'lulu_peanutbutter_510g', 'lux_soapradiantglow_150g', 'maggi_noodlesspicycheesy_75g', 'maggi_noodlesspicymanchow_244g', 'marilight_100g', 'marvella_strawberrycake_20g', 'milma_ghee_200ml', 'oreo_strawberry_43', 'paperboat_alphonso_mango_150ml', 'paperboat_jaljeera_200ml', 'parleg_biscut_50g', 'redbull_250ml', 'ripple_classicgreentea_50g', 'samrudhi_payasakutu_30g', 'snactac_tomatosoup_14g', 'sundrop_peanutbutter_510g', 'toonpops_10g', 'ujala_liquiddetergent_430ml', 'vivel_bodywash_200ml', 'waiwai_cupnoodles', 'weikfield_instantpastacheesymac_64g']
product_dict = {product : 60 for product in class_names_list}


def make_object_final(object_name,file_name) : #expiry_details.json(packed products)
    if os.path.exists(f"data/{file_name}"):
        # Read the existing data
        with open(f"data/{file_name}", 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []  # If file is empty or has invalid JSON
    else:
        print(f"the details file is not yet created")
        
    for index, entry in enumerate(data):
        if entry['object_name'] == object_name:
            new_object_name = object_name +'#'+str(time.time())
            entry['object_name'] = new_object_name
            
            if entry['expiry'] == "missing" and entry['mfg'] != "missing" :
                print("PRE EXP MISSING")
                year, month, day = entry['mfg'].split("/")
                ordinary_date = date(int(year),int(month),int(day))
                date_offset = product_dict[object_name]
                new_date = ordinary_date + timedelta(days=date_offset)
                new_date_string = new_date.isoformat()
                print(f"NEW EXPIRY : {new_date_string}")
                entry['expiry'] = new_date_string.replace("-","/")
            
            if entry['mfg'] == "missing" and entry['expiry'] != "missing" :
                print("PRE MFG MISSING")
                year, month, day = entry['expiry'].split("/")
                ordinary_date = date(int(year),int(month),int(day))
                date_offset = product_dict[object_name]
                new_date = ordinary_date - timedelta(days=date_offset)
                new_date_string = new_date.isoformat()
                print(f"NEW MFG : {new_date_string}")
                entry['mfg'] = new_date_string.replace("-","/")
            
            if entry['expiry'] != "missing" and entry['mfg'] != "missing" :
                year, month, day = entry['expiry'].split("/")
                exp_ordinary_date = date(int(year),int(month),int(day))

                year, month, day = entry['mfg'].split("/")
                mfg_ordinary_date = date(int(year),int(month),int(day))

                current_date = date.today()

                if exp_ordinary_date > current_date :
                    expired = "NO"
                    entry['expired'] = expired
                elif exp_ordinary_date < current_date :
                    expired = "YES"
                    entry['expired'] = expired
                
                
                if entry['expired'] != "YES" and entry['expired'] != "NULL" and entry['expired'] == "NO" :
                    life = (exp_ordinary_date - mfg_ordinary_date).days
                    entry['life'] = life
                 
            


    with open(f"data/{file_name}", 'w') as file:
        json.dump(data, file, indent=4)


def clear_list(file_name):
    if os.path.exists(f"data/{file_name}") :
        with open(f"data/{file_name}", 'w') as file:
                    data = []
                    json.dump(data, file, indent=4)
