import numpy as np
import copy
import functions as fn
import random
import pandas as pd


random.seed(2023)


def realization(inp, real_idoszakok, test):
    k = inp['x0'].shape[0]
    res = {}
    res['a'] = np.empty((1, real_idoszakok))
    res['ar'] = np.empty((1, real_idoszakok))
    res['term'] = np.empty((k, real_idoszakok))
    res['kapacitas_kihasznaltsag'] = np.empty((k, real_idoszakok))
    res['hitelarany_dont'] = np.empty((k, real_idoszakok))
    res['hitelarany_merleg'] = np.empty((k, real_idoszakok))
    res['csodvaloszinuseg'] = np.empty((k, real_idoszakok))
    res['fcfe'] = np.empty((k, real_idoszakok))
    for key in res.keys():
        res[key].fill(np.nan)
    res['elso_csod_ideje'] = 0

    for i in range(real_idoszakok):
        if i == 1:
            inp['szamlalo'] += 0
        if test:
            dont = inp['x0'][:]
        else:
            dont = fn.nash_egyensuly(inp)
        res['term'][:,i] = dont[:,0]
        res['hitelarany_dont'][:,i] = dont[:,2]
        if i == 0:
            hist_ocf_ts = fn.create_init_ocf_ts(res['term'][:,i], res['hitelarany_dont'][:,i],
                                                inp['a0'], inp['b'], inp['valt_kts_c'], inp['fix_kts_f'],
                                                inp['adokulcs'], inp['hitel_kamatlab'], inp['gepar'], inp['gepkap'])
            inp['ocf0'] = hist_ocf_ts[0]
            inp['ts0'] = hist_ocf_ts[1]
            hist_dont = fn.create_history(inp['gep_elettartam'], np.transpose([res['term'][:,i]]),
                                     np.transpose([res['hitelarany_dont'][:,i]]))
            hist_data = fn.calc_from_init_history(hist_dont[0], hist_dont[1], inp['gepkap'], inp['gep_elettartam'],
                                                  inp['gepar'], inp['futamido'], inp['hitel_kamatlab'],
                                                  inp['tervezes_idoszakok'])
            inp['meglevo_gep0'] = hist_data[0]
            inp['selejtezes_inp'] = hist_data[1]
            inp['amort_inp'] = hist_data[2]
            inp['hiteltorl_inp'] = hist_data[3]
            inp['kamatfiz_inp'] = hist_data[4]
            inp['eszkoz0'] = hist_data[5]
            inp['hitel0'] = hist_data[6]

        beruh_hitel = fn.calc_beruh_hitel(np.transpose([res['term'][:, i]]), dont[:,1],
                                          np.transpose([res['hitelarany_dont'][:, i]]), inp['gepkap'],
                                          inp['gep_elettartam'], inp['gepar'], inp['futamido'], inp['hitel_kamatlab'],
                                          inp['meglevo_gep0'], inp['selejtezes_inp'], inp['amort_inp'],
                                          inp['hiteltorl_inp'], inp['kamatfiz_inp'])
        meglevo_gepek = beruh_hitel[7][:]
        res['kapacitas_kihasznaltsag'][:,i] = res['term'][:,i] / (meglevo_gepek * inp['gepkap'])
        beruh = beruh_hitel[0]
        hitelfelv = beruh_hitel[1]
        eszkoz = inp['eszkoz0'] + beruh - inp['amort0']
        hitel = inp['hitel0'] + hitelfelv - inp['hiteltorl0']
        res['hitelarany_merleg'][:,i] = hitel / eszkoz

        fcff = inp['ocf0'] - beruh
        d_hitel = hitelfelv - inp['hiteltorl0']
        if i == 0:
            d_hitel = np.zeros(k)  # feltétel, hogy eddig egyensúlyban volt
            inp['kamatfiz0'] = hitel * inp['hitel_kamatlab']
        res['fcfe'][:,i] = fcff + inp['ts0'] + d_hitel - inp['kamatfiz0']
        if np.any(res['fcfe'][:,i] < 0):
            res['elso_csod_ideje'] = i
            break

        # új év
        # csődvalószínűség
        # TODO: átírni vektorokra
        kamatfiz = beruh_hitel[6][:,0]
        amort = beruh_hitel[3][:,0]
        hiteltorl = beruh_hitel[5][:,0]
        for j in range(k):
            term = np.array([res['term'][:,i][j]])
            term_dont_others = res['term'][:,i][np.arange(len(res['term'][:,i]))!=j]
            if not np.any(term_dont_others):
                term_others = np.zeros(1)
            else:
                term_others = np.array([np.sum(term_dont_others)])
            csodval = 1 - fn.calc_fcff_ts(term, term_others, inp, inp['a0'], np.array([kamatfiz[j]]), np.array([beruh[j]]),
                                          np.array([amort[j]]), np.array([hitelfelv[j]]), np.array([hiteltorl[j]]))[2]
            res['csodvaloszinuseg'][j,i] = csodval
        # rnumber = random.uniform(0,1)
        rnumber = veletlenek_sorozata[i]
        if rnumber < inp['p'][0]:
            res['a'][0,i] = inp['a0'] * (1 + fn.get_valt(inp['trend'], inp['szoras'])[0])
        elif rnumber < inp['p'][0] + inp['p'][1]:
            res['a'][0,i] = inp['a0'] * (1 + fn.get_valt(inp['trend'], inp['szoras'])[1])
        else:
            res['a'][0,i] = inp['a0'] * (1 + fn.get_valt(inp['trend'], inp['szoras'])[2])
        res['ar'][0,i] = res['a'][0,i] - inp['b'] * np.sum(res['term'][:,i])
        # ocf
        arbevetel = res['term'][:,i] * res['ar'][0,i]
        valt_kts = res['term'][:,i] * inp['valt_kts_c']
        fix_kts = inp['fix_kts_f'] * np.ones(k)
        kamatfiz = beruh_hitel[6][:,0]
        amort = beruh_hitel[3][:,0]
        ebit = arbevetel - valt_kts - fix_kts - amort
        uzemi_eredmeny_adoja = np.maximum(ebit * inp['adokulcs'], np.zeros(k))
        inp['ocf0'] = ebit - uzemi_eredmeny_adoja + amort
        ado = np.maximum((ebit - kamatfiz) * inp['adokulcs'], np.zeros(k))
        inp['ts0'] = uzemi_eredmeny_adoja - ado

        # többi változó átírása
        inp['a0'] = res['a'][0,i]
        inp['meglevo_gep0'] = meglevo_gepek
        inp['selejtezes_inp'] = np.hstack((beruh_hitel[8][:,1:], np.transpose([np.zeros(k)])))
        inp['amort_inp'] = np.hstack((beruh_hitel[3][:,1:], np.transpose([np.zeros(k)])))
        inp['hiteltorl_inp'] = np.hstack((beruh_hitel[5][:,1:], np.transpose([np.zeros(k)])))
        inp['kamatfiz_inp'] = np.hstack((beruh_hitel[6][:,1:], np.transpose([np.zeros(k)])))
        inp['eszkoz0'] = eszkoz[:]
        inp['hitel0'] = hitel[:]
        inp['amort0'] = amort[:]
        inp['hiteltorl0'] = hiteltorl[:]
        inp['kamatfiz0'] = kamatfiz[:]
        inp['elsofutas'] = False
        inp['x0'] = copy.deepcopy(dont)
    return res


for vallalatok_szama in range(1, 3):
    for scenario in range(2):
        if scenario == 0:
            sctext = "fel-le"
            veletlenek_sorozata = [0.5, 0, 0.5, 0, 0.5, 1, 0, 0.5, 1, 0.5, 0.5, 1, 1, 0, 0.5]  # fel, le
        else:
            sctext = "le-fel"
            veletlenek_sorozata = [0.5, 1, 0.5, 1, 0.5, 0, 1, 0.5, 0, 0.5, 0.5, 0, 0, 1, 0.5]  # le, fel
        real_idoszakok = 15
        # veletlenek_sorozata = [0.5, 1, 0.5, 1, 0.5, 0, 1, 0.5, 0, 0.5, 0.5, 0, 0, 1, 0.5]  # le, fel
        # veletlenek_sorozata = [0.5, 0, 0.5, 0, 0.5, 1, 0, 0.5, 1, 0.5, 0.5, 1, 1, 0, 0.5]  # fel, le
        # vallalatok_szama = 2
        # inputok
        inp = {}
        # debug
        inp['szamlalo'] = 0
        # állandók
        inp['tervezes_idoszakok'] = 6
        inp['trend'] = 0
        inp['szoras'] = 0.02
        inp['p'] = np.array([0.25, 0.5, 0.25])
        inp['b'] = 0.002
        inp['elvart_hozam'] = 0.15
        inp['adokulcs'] = 0.2
        inp['gep_elettartam'] = 5
        inp['gepkap'] = 5
        inp['gepar'] = 80
        inp['futamido'] = 5
        inp['hitel_kamatlab'] = 0.12
        inp['valt_kts_c'] = 10
        inp['fix_kts_f'] = 1050
        # változók
        inp['a0'] = 24.0
        inp['meglevo_gep0'] = np.zeros(vallalatok_szama)
        inp['selejtezes_inp'] = np.empty((vallalatok_szama, inp['tervezes_idoszakok']))
        inp['amort_inp'] = np.empty((vallalatok_szama, inp['tervezes_idoszakok']))
        inp['hiteltorl_inp'] = np.empty((vallalatok_szama, inp['tervezes_idoszakok']))
        inp['kamatfiz_inp'] = np.empty((vallalatok_szama, inp['tervezes_idoszakok']))
        inp['eszkoz0'] = np.zeros(vallalatok_szama)
        inp['hitel0'] = np.zeros(vallalatok_szama)
        inp['amort0'] = np.zeros(vallalatok_szama)
        inp['hiteltorl0'] = np.zeros(vallalatok_szama)
        inp['ocf0'] = np.zeros(vallalatok_szama)
        inp['ts0'] = np.zeros(vallalatok_szama)
        inp['kamatfiz0'] = np.zeros(vallalatok_szama)
        inp['elsofutas'] = True
        inp['x0'] = np.tile(np.array([1000 / vallalatok_szama, 0, 0.5]), (vallalatok_szama, 1))

        r = realization(inp, real_idoszakok, False)
        rownames = list()
        data = np.zeros((1, real_idoszakok))
        for key in r.keys():
            if key != 'elso_csod_ideje':
                data = np.vstack((data, r[key]))
                if r[key].shape[0] == 1:
                    rownames.append(key)
                else:
                    k = r[key].shape[0]
                    for i in range(k):
                        rownames.append(key + "_" + str(i + 1))
        data = np.delete(data, 0, 0)
        df = pd.DataFrame(data, index=rownames, columns=range(1, real_idoszakok + 1))
        df.to_excel(str(vallalatok_szama) + "-vállalat-opt-bizonytalanság_" + sctext + "_v0.2.xlsx")
        # df.to_excel("2(-ugyanolyan)-vállalat-opt-nincs-bizonytalanság.xlsx")