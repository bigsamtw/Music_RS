import os 
import tqdm
import math
import random
import operator
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm_notebook

dir_ = '../data/'
file_name = 'normalized_to_rating_filter_track_5_user_100.csv'

train = pd.read_pickle(os.path.join(dir_, 'train_' + file_name[:-3] + 'pkl'))
test = pd.read_pickle(os.path.join(dir_, 'test_' + file_name[:-3] + 'pkl'))

predictions_user = pd.read_pickle(os.path.join(dir_, 'prediction_cf_user_top_N_' + file_name[:-3] + 'pkl'))
predictions_SVD_user = pd.read_pickle(os.path.join(dir_, 'prediction_SVD_user_top_N_' + file_name[:-3] + 'pkl'))
prediction_SVD = pd.read_pickle(os.path.join(dir_, 'prediction_svd_top_N_' + file_name[:-3] + 'pkl'))
prediction_popularity = pd.read_pickle(os.path.join(dir_, 'prediction_popularity_count_top_N_normalized_popularity_filter_track_5_user_100.pkl'))
prediction_popularity = prediction_popularity.sort_values(by=['count'],  ascending=False)

time_to_sim = 120
num_user = len(train['uid'].unique())

def simulation(q,no):
    time_sim = 0
    user_status = []
    for i in range(num_user):
        user_status.append(False)

    customers = []
    customer_init = int(random.random()*10//1 + 5) # 5-15
    while customer_init > 0:
        group_size = int(random.random()*5//1 + 1)
        if group_size > customer_init:
            group_size = customer_init # Constrain the value
                    
        customer = []
        
        for i in range(group_size):
            u = int(random.random()*num_user//1)
            if user_status[u] == False:
                user_status[u] = True
            else:
                while user_status[u]:
                    u = int(random.random()*num_user//1)
                user_status[u] = True
            customer.append(u)
            
        customers.append(customer)
        customer_init -= group_size

    arrivals = []
    for i in range(len(customers)):
        arrivals.append(0)

    while time_sim < time_to_sim:
        
        inter_arrival = int(np.random.exponential(5))
        arrival = time_sim + inter_arrival
        
        if arrival < time_to_sim:
            customer = []        
            group_size = int(random.random()*5//1 + 1)

            for i in range(group_size):
                u = int(random.random()*num_user//1)
                if user_status[u] == False:
                    user_status[u] = True
                else:
                    while user_status[u]:
                        u = int(random.random()*num_user//1)
                    user_status[u] = True
                customer.append(u)

            customers.append(customer)
            arrivals.append(arrival)
            
        time_sim += inter_arrival

    departures = []
    for i in range(len(customers)):
        service_time = int(np.random.exponential(20))
        if service_time < 5:
            service_time = 5
        departure = arrivals[i] + service_time
        departures.append(departure)

    pd_sim = pd.DataFrame(columns=['gid','arrival','departure', 'sevice', 'members'])

    for i in range(len(customers)):
        pd_sim = pd_sim.append({'gid': i, 'arrival': arrivals[i], 'departure': departures[i],'sevice': (departures[i]-arrivals[i]), 'members': customers[i]}, ignore_index=True)

    n = 10000
    Top_N_List =  pd.DataFrame() 
    for _, group in pd_sim.iterrows():
        if len(group['members']) == 1:
            m = prediction_SVD[prediction_SVD['uid'] == group['members'][0]]
        elif len(group['members']) == 2:
            u1 = prediction_SVD[prediction_SVD['uid'] == group['members'][0]]
            u2 = prediction_SVD[prediction_SVD['uid'] == group['members'][1]]
            m = pd.merge(u1, u2, on=['tid'], how='inner', suffixes=['_u1','_u2'])
            m = m[['tid', 'rating_u1','rating_u2']]
            m['rating'] = (m['rating_u1'] + m['rating_u2'])/2
        elif len(group['members']) == 3:
            u1 = prediction_SVD[prediction_SVD['uid'] == group['members'][0]]
            u2 = prediction_SVD[prediction_SVD['uid'] == group['members'][1]]
            u3 = prediction_SVD[prediction_SVD['uid'] == group['members'][2]]
            m = pd.merge(u1, u2, on=['tid'], how='inner', suffixes=['_u1','_u2'])
            m = pd.merge(m,  u3, on=['tid'], how='inner')
            m = m[['tid', 'rating_u1','rating_u2','rating']]
            m['rating'] = (m['rating_u1'] + m['rating_u2'] + m['rating'])/3
        elif len(group['members']) == 4:
            u1 = prediction_SVD[prediction_SVD['uid'] == group['members'][0]]
            u2 = prediction_SVD[prediction_SVD['uid'] == group['members'][1]]
            u3 = prediction_SVD[prediction_SVD['uid'] == group['members'][2]]
            u4 = prediction_SVD[prediction_SVD['uid'] == group['members'][3]]
            m = pd.merge(u1, u2, on=['tid'], how='inner', suffixes=['_u1','_u2'])
            m = pd.merge(m,  u3, on=['tid'], how='inner', suffixes=['','_u3'])
            m = pd.merge(m,  u4, on=['tid'], how='inner', suffixes=['_u3','_u4'])
            m = m[['tid', 'rating_u1','rating_u2','rating_u3','rating_u4']]
            m['rating'] = (m['rating_u1'] + m['rating_u2'] + m['rating_u3'] + m['rating_u4'])/4
        elif len(group['members']) == 5:
            u1 = prediction_SVD[prediction_SVD['uid'] == group['members'][0]]
            u2 = prediction_SVD[prediction_SVD['uid'] == group['members'][1]]
            u3 = prediction_SVD[prediction_SVD['uid'] == group['members'][2]]
            u4 = prediction_SVD[prediction_SVD['uid'] == group['members'][3]]
            u5 = prediction_SVD[prediction_SVD['uid'] == group['members'][4]]
            m = pd.merge(u1, u2, on=['tid'], how='inner', suffixes=['_u1','_u2'])
            m = pd.merge(m,  u3, on=['tid'], how='inner', suffixes=['','_u3'])
            m = pd.merge(m,  u4, on=['tid'], how='inner', suffixes=['','_u4'])
            m = pd.merge(m,  u5, on=['tid'], how='inner', suffixes=['_u3','_u5'])
            m = m[['tid', 'rating_u1','rating_u2','rating_u3','rating_u4','rating_u5']]
            m['rating'] = (m['rating_u1'] + m['rating_u2'] + m['rating_u3'] + m['rating_u4'] + m['rating_u5'])/5 
        else:
            print('Error')
        m = m[['tid', 'rating']]
        m = m.sort_values(by=['rating'],  ascending=False)
        m = m[:n]   
        m['gid'] = group['gid']
        Top_N_List = Top_N_List.append(m)

    #____________________________Evaluation Static____________________________________________________________#
    result = []

    total_all = 0
    hit_SVD_all = 0
    hit_pop_all = 0
    rating = 0
    for _, group in pd_sim.iterrows():
        gid = group['gid']
        members = group['members']
        Top_N = Top_N_List[Top_N_List['gid'] == gid].copy()
        Top_N = Top_N.sort_values(by=['rating'],  ascending=False)
        
        total = 0
        hit_SVD = 0
        hit_pop = 0
        for t in range(4):
            total += 1
            tid = Top_N.iloc[t]['tid']
            for uid in members:
                test_SVD = test[test['uid'] == uid].copy()
                test_SVD = test_SVD[test_SVD['tid'] == tid]
                if len(test_SVD) > 0 and test_SVD.iloc[0]['rating'] >= rating:
                    hit_SVD +=1
                    break
        for t in range(4):
            tid = prediction_popularity.iloc[t]['tid'].copy()
            for uid in members:
                test_pop = test[test['uid'] == uid].copy()
                test_pop = test_pop[test_pop['tid'] == tid]
                if len(test_pop) > 0 and test_pop.iloc[0]['rating'] >= rating:
                    hit_pop +=1
                    break
        total_all += total
        hit_SVD_all += hit_SVD
        hit_pop_all += hit_pop
    
    total_all = 0
    sat_SVD_all = 0
    sat_pop_all = 0
    for _, group in pd_sim.iterrows():
        gid = group['gid']
        members = group['members']
        Top_N = Top_N_List[Top_N_List['gid'] == gid].copy()
        Top_N = Top_N.sort_values(by=['rating'],  ascending=False)
        
        total = 0
        sat_SVD = 0
        sat_pop = 0
        for t in range(4):
            hit_SVD = 0
            total += 1
            tid = Top_N.iloc[t]['tid']
            for uid in members:
                test_SVD = test[test['uid'] == uid].copy()
                test_SVD = test_SVD[test_SVD['tid'] == tid]
                if len(test_SVD) > 0 and test_SVD.iloc[0]['rating'] >= rating:
                    hit_SVD +=1
            sat_SVD += math.log(hit_SVD+1,10)/math.log(len(members)+1,10)
            
        for t in range(4):
            hit_pop = 0
            tid = prediction_popularity.iloc[t]['tid'].copy()
            for uid in members:
                test_pop = test[test['uid'] == uid].copy()
                test_pop = test_pop[test_pop['tid'] == tid]
                if len(test_pop) > 0 and test_pop.iloc[0]['rating'] >= rating:
                    hit_pop +=1
            sat_pop += math.log(hit_pop+1,10)/math.log(len(members)+1,10)
        total_all += total
        sat_SVD_all += sat_SVD
        sat_pop_all += sat_pop

    result.append(hit_SVD_all/total_all)
    result.append(hit_pop_all/total_all)
    result.append(sat_SVD_all/total_all)
    result.append(sat_pop_all/total_all)
    
    #____________________________End of Evaluation Static____________________________________________________________#

    Top_N_List['listened'] = False
    Top_N_List['count'] = 1
    Top_N_List

    playlist = []
    for i in range(int(time_to_sim/5)):
        groups = pd_sim[pd_sim['arrival']<=i*5]
        groups = groups[groups['departure']>i*5]
        rating_table = pd.DataFrame() 
        for _, j in groups.iterrows():
            if len(rating_table) == 0:
                rating_table = Top_N_List[Top_N_List['gid'] == j['gid']].copy()
            else:
                r_t = Top_N_List[Top_N_List['gid'] == j['gid']].copy()
                rating_table = rating_table.set_index('tid').add(r_t.set_index('tid'), fill_value=0).reset_index()
            rating_table = rating_table.sort_values(by=['rating'],  ascending=False)
        
        if len(rating_table) > 0:            
            tid = rating_table.iloc[0]['tid']
            k = 0
            while rating_table.iloc[k]['listened'] and rating_table.iloc[k]['count'] < len(groups): #避免重複推薦
                k += 1
                tid = rating_table.iloc[k]['tid']
            playlist.append(tid)
            
            for _, j in groups.iterrows(): #避免重複推薦(紀錄已經被推薦過的歌曲)
                gid = j['gid']
                Top_N_List.loc[operator.and_((Top_N_List['tid']==tid), (Top_N_List['gid']==gid)), ['listened']] = True
        else:
            playlist.append(-1)


    playlist_pop = []
    pop_list = np.zeros(24)
    pop = prediction_popularity[:24].reset_index().copy()
    for i in range(int(time_to_sim/5)):
        t = int(random.random()*24//1)
        while pop_list[t] != 0:
            t = int(random.random()*24//1)
        tid = pop.iloc[t]['tid']
        playlist_pop.append(tid)
        pop_list[t] += 1

    # playlist_pop = []
    # prediction_popularity = prediction_popularity.sort_values(by=['count'],  ascending=False)
    # pop = prediction_popularity[:100].reset_index().copy()
    # for i in range(int(time_to_sim/5)):
    #     t = int(random.random()*100//1)
    #     tid = pop.iloc[t]['tid']
    #     playlist_pop.append(tid)

    #____________________________Evaluation____________________________________________________________#
    

    totals = []
    hits_SVD = []
    hits_pop = []
        
    for _, group in pd_sim.iterrows():
        arrival = group['arrival']
        departure = group['departure']
        members = group['members']
        start = int(arrival//5)
        end = int(math.ceil(departure/5))
        if end > len(playlist):
            end = len(playlist)
        
        total = 0
        hit_SVD = 0
        hit_pop = 0
        for t in range(start,end):
            total += 1
            for uid in members:
                test_SVD = test[test['uid'] == uid].copy()
                test_SVD = test_SVD[test_SVD['tid'] == playlist[t]]
                if len(test_SVD) > 0:
                    hit_SVD +=1
                    break
        for t in range(start,end):
            for uid in members:
                test_pop = test[test['uid'] == uid].copy()
                if playlist_pop[t] == -1:
                    print('Error')
                test_pop = test_pop[test_pop['tid'] == playlist_pop[t]]
                if len(test_pop) > 0:
                    hit_pop +=1
                    break
        totals.append(total)
        hits_SVD.append(hit_SVD)
        hits_pop.append(hit_pop)

    precision_SVD = []
    precision_pop = []
    for i in range(len(pd_sim)):
        precision_SVD.append(hits_SVD[i]/totals[i])
        precision_pop.append(hits_pop[i]/totals[i])

    avg_precision_SVD = 0
    avg_precision_pop = 0
    for i in range(len(pd_sim)):
        avg_precision_SVD += precision_SVD[i]
        avg_precision_pop += precision_pop[i]
    avg_precision_SVD /= len(pd_sim)
    avg_precision_pop /= len(pd_sim)

    totals = []
    satisfactions_SVD = []
    satisfactions_pop = []
        
    for _, group in pd_sim.iterrows():
        arrival = group['arrival']
        departure = group['departure']
        members = group['members']
        start = int(arrival//5)
        end = int(math.ceil(departure/5))
        if end > len(playlist):
            end = len(playlist)
        
        total = 0

        satisfaction_SVD = 0
        satisfaction_pop = 0
        for t in range(start,end):
            total += 1
            hit_SVD = 0
            for uid in members:
                test_SVD = test[test['uid'] == uid].copy()
                test_SVD = test_SVD[test_SVD['tid'] == playlist[t]]
                if len(test_SVD) > 0:
                    hit_SVD +=1
            satisfaction_SVD += math.log(hit_SVD+1,10)/math.log(len(members)+1,10)
            
        for t in range(start,end):
            hit_pop = 0
            for uid in members:
                test_pop = test[test['uid'] == uid].copy()
                test_pop = test_pop[test_pop['tid'] == playlist_pop[t]]
                if len(test_pop) > 0:
                    hit_pop +=1
            satisfaction_pop += math.log(hit_pop+1,10)/math.log(len(members)+1,10)
        totals.append(total)
        satisfactions_SVD.append(satisfaction_SVD)
        satisfactions_pop.append(satisfaction_pop)

    for i in range(len(pd_sim)):
        satisfactions_SVD[i] = satisfactions_SVD[i]/totals[i]
        satisfactions_pop[i] = satisfactions_pop[i]/totals[i]

    avg_satisfactions_SVD = 0
    avg_satisfactions_pop = 0
    for i in range(len(pd_sim)):
        avg_satisfactions_SVD += satisfactions_SVD[i]
        avg_satisfactions_pop += satisfactions_pop[i]
    avg_satisfactions_SVD /= len(pd_sim)
    avg_satisfactions_pop /= len(pd_sim)
    result.append(avg_precision_SVD)
    result.append(avg_precision_pop)
    result.append(avg_satisfactions_SVD)
    result.append(avg_satisfactions_pop)
    
    #_____________________________________End of Evaluation________________________________________________________________#


    # customers_in_store = []
    # for i in range(int(time_to_sim/5)):
    #     groups = pd_sim[pd_sim['arrival']<=i*5]
    #     groups = groups[groups['departure']>i*5]
    #     customers_in_store.append(len(groups))

    # group_size = np.zeros(5)
    # for _, i in pd_sim.iterrows():
    #     group_size[len(i['members'])-1] += 1

    # service_time = np.zeros(24)
    # for _, i in pd_sim.iterrows():
    #     service_time[int(int(i['sevice'])//10)] += 1

    # print(customers_in_store)
    # print(group_size)
    # print(service_time)
    print(result[:4], '\n', result[4:8])
    q.put(result)

num_sim = 100

if __name__=='__main__':
    r = []
    sim = 0
    while sim < num_sim:
        q1 = mp.Queue()   # 使用 queue 接收 function 的回傳值
        q2 = mp.Queue()   # 使用 queue 接收 function 的回傳值
        q3 = mp.Queue()   # 使用 queue 接收 function 的回傳值
        q4 = mp.Queue()   # 使用 queue 接收 function 的回傳值
        q5 = mp.Queue()   # 使用 queue 接收 function 的回傳值
        p1 = mp.Process(target=simulation, args=(q1,1)) # 特別注意 這邊的傳入參數只有一個的話，後面要有逗號
        p2 = mp.Process(target=simulation, args=(q2,2))
        p3 = mp.Process(target=simulation, args=(q3,3)) 
        p4 = mp.Process(target=simulation, args=(q4,4))
        p5 = mp.Process(target=simulation, args=(q5,5)) 

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()

        r.append(q1.get())        
        r.append(q2.get())        
        r.append(q3.get())        
        r.append(q4.get())        
        r.append(q5.get())        

        
        sim += 5
    #____________Calculation____________#
    static_avg_p_SVD = 0
    static_avg_p_pop = 0
    static_avg_s_SVD = 0
    static_avg_s_pop = 0
    for r_ in r:
        static_avg_p_SVD += r_[0]
        static_avg_p_pop += r_[1]
        static_avg_s_SVD += r_[2]
        static_avg_s_pop += r_[3]

    static_avg_p_SVD /= num_sim
    static_avg_p_pop /= num_sim
    static_avg_s_SVD /= num_sim
    static_avg_s_pop /= num_sim

    print(static_avg_p_SVD, static_avg_p_pop, static_avg_s_SVD, static_avg_s_pop)

    avg_p_SVD = 0
    avg_p_pop = 0
    avg_s_SVD = 0
    avg_s_pop = 0
    for r_ in r:
        avg_p_SVD += r_[4]
        avg_p_pop += r_[5]
        avg_s_SVD += r_[6]
        avg_s_pop += r_[7]

    avg_p_SVD /= num_sim
    avg_p_pop /= num_sim
    avg_s_SVD /= num_sim
    avg_s_pop /= num_sim

    print(avg_p_SVD, avg_p_pop, avg_s_SVD, avg_s_pop)
    
    #____________End of Calculation____________#

    np.savetxt('result_static_dynamic', r, delimiter=",")
