import numpy as np
import cloudpickle
#import pickle
rf_classifer = cloudpickle.load(open("./automl_cls_2018_era5_ex_elv.pkl", "rb"))
rf_regressor = cloudpickle.load(open("./automl_reg_2018_era5_ex_elv.pkl", "rb"))
def pred_automl(x):
       value_fine = rf_regressor.predict(x)
       mask_fine = rf_classifer.predict(x)
       value_fine = 10**(value_fine)-1  # log-trans
       value_fine[mask_fine==0] = 0
       value_fine[value_fine<0] = 0
       return value_fine

if  __name__ == "__main__":
    # t2m_fine, sp_fine, q_fine, strd_fine, ssrd_fine, ws_fine, lat_fine, lon_fine, elev_fine
    t2m_f=[295]
    sp_f=[99000]
    q_f=[0.1]
    strd_f=[100]
    ssrd_f=[100]
    ws_f=[5]
    elev_f=[100]
    #    x = np.concatenate([t2m_f,sp_f,q_f,strd_f,ssrd_f,ws_f,elev_f],axis=-1)
    x = np.array([t2m_f,sp_f,q_f,strd_f,ssrd_f,ws_f,elev_f]).T#.reshape(1, -1)
    precp = pred_automl(x)
    print(precp)
