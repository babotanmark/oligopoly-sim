import numpy as np
from scipy import optimize
import copy
import time


vallalatok_szama = 1
# inputok
inp = {}
# debug
inp['szamlalo'] = 0
# állandók
inp['tervezes_idoszakok'] = 6
inp['trend'] = 0
inp['szoras'] = 0
inp['p'] = np.array([1 / 3, 1 / 3, 1 / 3])
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
inp['a0'] = 21
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
inp['x0'] = np.tile(np.array([1000/vallalatok_szama, 0, 0.5]), (vallalatok_szama, 1))
#
# döntési változók
term_szint_dont = 1000
g_dont = 0
hitelarany_dont = 0.5
dont = np.array([term_szint_dont, g_dont, hitelarany_dont])


def get_term_vector(term_szint, g, n):
    if isinstance(term_szint, float):
        res = np.empty((1,n))
    else:
        res = np.empty((len(term_szint),n))
    for i in range(n):
        res[:,i] = term_szint * (1 + g) ** i
    return res


def create_history(elettart, term_szint, hitelarany):
    """Az első döntésnél a múlt felírása
    Csak a termelési és hitelfevételi döntés"""
    alap = term_szint / elettart
    term_hist = alap * np.arange(1, elettart)
    hitelarany_hist = hitelarany * np.ones(elettart - 1)
    return [term_hist, hitelarany_hist]


# print(create_history(inp['gep_elettartam'], term_szint_dont, hitelarany_dont))

def create_init_ocf_ts(term_szint, hitelarany, a, b, c, fix, adokulcs, kamatlab, gepar, gepkap):
    ocf_res = ((a - b * term_szint - c) * term_szint - fix) * (1 - hitelarany * adokulcs)
    eszkoz = np.ceil(term_szint / gepkap) * gepar
    hitel = eszkoz * hitelarany
    adopajzs_res = hitel * kamatlab * adokulcs
    return [ocf_res, adopajzs_res]


# init_ocf_ts = create_init_ocf_ts(term_szint_dont, hitelarany_dont, inp['a0'], inp['b'], inp['valt_kts_c'],
#                                  inp['fix_kts_f'], inp['adokulcs'], inp['hitel_kamatlab'], inp['gepar'], inp['gepkap'])
# ocf0 = init_ocf_ts[0]
# ts0 = init_ocf_ts[1]


def calc_from_init_history(term_hist, hitelarany_hist, gepkap, elettart, gepar, futamido, kamatlab, n):
    """A múltból a fix adottságok felírása
    0: meglévő gépek a 0. évben
    1: selejtezés (vektor 0-(n-1))
    2: amortizáció (vektor)
    3: hiteltörlesztés (vektor)
    4: kamatfizetés (vektor)
    5: ic_0
    6: hitel_0"""
    szukseges_gepek = np.ceil(term_hist / gepkap)
    if isinstance(szukseges_gepek[-1], np.ndarray):
        meglevo_gep_res = szukseges_gepek[:,-1]
        k = len(meglevo_gep_res)
        meglevo_gepek = np.hstack((np.transpose([np.zeros(k)]), szukseges_gepek[:,:-1]))
        gepvasarlas = np.maximum(szukseges_gepek - meglevo_gepek, np.zeros(szukseges_gepek.shape))
        max_index = min(elettart, n)
        selejtezes_res = np.zeros((k, n))
        selejtezes_res[:,1:max_index] = gepvasarlas
        beruh = gepvasarlas * gepar
        amort_seged = beruh / elettart
        amort_res = np.zeros((k, n))
        for i in range(max_index):
            amort_res[:,i] = np.sum(amort_seged[:,i:], axis=1)
        hitelfelv = beruh * hitelarany_hist
        max_index = min(futamido, n)
        hiteltorl_res = np.zeros((k, n))
        hiteltorl_res[:,:max_index-1] = hitelfelv
        eves_kamat = hitelfelv * kamatlab
        kamatfiz_res = np.zeros((k, n))
        for i in range(max_index):
            kamatfiz_res[:,i] = np.sum(eves_kamat[:,i:], axis=1)
        ic0_res = szukseges_gepek[:,-1] * gepar
        hitel0_res = ic0_res * hitelarany_hist[:,-1]
    else:
        meglevo_gep_res = szukseges_gepek[-1]
        meglevo_gepek = np.insert(szukseges_gepek[:-1], 0, 0)
        gepvasarlas = np.maximum(szukseges_gepek - meglevo_gepek, np.zeros(len(szukseges_gepek)))
        max_index = min(elettart, n)  # inp['gep_elettartam'] = len(gepvasarlas)+1
        selejtezes_res = np.zeros((1, n))
        selejtezes_res[0,1:max_index] = gepvasarlas
        beruh = gepvasarlas * gepar
        amort_seged = beruh / elettart
        amort_res = np.zeros((1, n))
        for i in range(max_index):
            amort_res[0, i] = np.sum(amort_seged[i:])
        hitelfelv = beruh * hitelarany_hist
        max_index = min(futamido, n)
        hiteltorl_res = np.zeros((1,n))
        hiteltorl_res[0,:max_index - 1] = hitelfelv
        eves_kamat = hitelfelv * kamatlab
        kamatfiz_res = np.zeros((1,n))
        for i in range(max_index):
            kamatfiz_res[0,i] = np.sum(eves_kamat[i:])
        ic0_res = szukseges_gepek[-1] * gepar
        hitel0_res = ic0_res * hitelarany_hist[-1]
    return [meglevo_gep_res, selejtezes_res, amort_res, hiteltorl_res, kamatfiz_res, ic0_res, hitel0_res]


# r = create_history(inp['gep_elettartam'], term_szint_dont, hitelarany_dont)
# term_hist = r[0]
# hitelarany_hist = r[1]
# r1 = calc_from_init_history(term_hist, hitelarany_hist, inp['gepkap'], inp['gep_elettartam'], inp['gepar'],
#                             inp['futamido'], inp['hitel_kamatlab'], inp['tervezes_idoszakok'])


# print(r1)


# ezek futnak le mindig
def calc_beruh_hitel(term, g, hitelarany, gepkap, elettart, gepar, futamido, kamatlab,
                     meglevo_gep0, selejtezes_inp, amort_inp, hiteltorl_inp, kamatfiz_inp):
    """Az Excel felső része
    beruh0 és beruh külön
    0: beruh0_res
    1: hitelfelv0_res
    2: beruh_res
    3: amort_res
    4: hitelfelv_res
    5: hiteltorl_res
    6: kamatfiz_res
    7: meglevo_gep0_res
    8: selejtezes"""
    if isinstance(term[0], np.ndarray):
        k,n = term.shape  # n==1 feltételezés
        szukseges_gepek = np.squeeze(np.ceil(term / gepkap), axis=1)
        selejtezes = copy.deepcopy(selejtezes_inp)
        gepvasarlas = np.maximum(szukseges_gepek - meglevo_gep0 + np.squeeze(selejtezes[:,0]), np.zeros(k))
        meglevo_gep0_res = meglevo_gep0 - np.squeeze(selejtezes[:,0]) + gepvasarlas
        selejtezes[:,elettart] += gepvasarlas
        beruh = gepvasarlas * gepar
        amort_res = copy.deepcopy(amort_inp)
        eves_amort = beruh / elettart
        for j in range(elettart):
            amort_res[:,j] += eves_amort
        hitelfelv = beruh * np.squeeze(hitelarany)
        hiteltorl_res = copy.deepcopy(hiteltorl_inp)
        hiteltorl_res[:,futamido-1] += hitelfelv
        kamatfiz_res = copy.deepcopy(kamatfiz_inp)
        eves_kamat = hitelfelv * kamatlab
        for j in range(futamido):
            kamatfiz_res[:,j] += eves_kamat
        beruh0_res = beruh[:]
        beruh_res = []
        hitelfelv0_res = hitelfelv[:]
        hitelfelv_res = []
    else:
        n = len(term)
        szukseges_gepek = np.ceil(term / gepkap)
        meglevo_gep0_res = szukseges_gepek[0]
        meglevo_gepek = np.insert(szukseges_gepek[:-1], 0, meglevo_gep0)
        if len(selejtezes_inp.shape) > 1:
            selejtezes = np.squeeze(selejtezes_inp)
        else:
            selejtezes = copy.deepcopy(selejtezes_inp)
        gepvasarlas = np.zeros(n)
        for i in range(n):
            gepvasarlas[i] = max(szukseges_gepek[i] - meglevo_gepek[i] + selejtezes[i], 0)
            if i + elettart < n:
                selejtezes[i + elettart] = gepvasarlas[i]
        beruh = gepvasarlas * gepar
        if len(amort_inp.shape) > 1:
            amort_res = np.squeeze(amort_inp)
        else:
            amort_res = copy.deepcopy(amort_inp)
        for i in range(n):
            eves_amort = beruh[i] / elettart
            max_index = min(i + elettart, n)
            for j in range(i, max_index):
                amort_res[j] += eves_amort
        hitelfelv = beruh * hitelarany
        if len(hiteltorl_inp.shape) > 1:
            hiteltorl_res = np.squeeze(hiteltorl_inp)
        else:
            hiteltorl_res = copy.deepcopy(hiteltorl_inp)
        hiteltorl_res[futamido - 1:] = hitelfelv[:n - futamido + 1]
        if len(kamatfiz_inp.shape) > 1:
            kamatfiz_res = np.squeeze(kamatfiz_inp)
        else:
            kamatfiz_res = copy.deepcopy(kamatfiz_inp)
        for i in range(n):
            eves_kamat = hitelfelv[i] * kamatlab
            max_index = min(i + futamido, n)
            for j in range(i, max_index):
                kamatfiz_res[j] += eves_kamat
        beruh0_res = beruh[0]
        beruh_res = np.append(beruh[1:], beruh[-1] * (1 + g))
        hitelfelv0_res = hitelfelv[0]
        hitelfelv_res = np.append(hitelfelv[1:], hitelfelv[-1] * (1 + g))
    return [beruh0_res, hitelfelv0_res, beruh_res, amort_res, hitelfelv_res, hiteltorl_res, kamatfiz_res,
            meglevo_gep0_res, selejtezes]


# term = get_term_vector(term_szint_dont, g_dont, inp['tervezes_idoszakok'])
# meglevo_gep0 = r1[0]
# selejtezes_inp = r1[1]
# amort_inp = r1[2]
# hiteltorl_inp = r1[3]
# kamatfiz_inp = r1[4]
# eszkoz0 = r1[5]
# hitel0 = r1[6]


# r2 = calc_beruh_hitel(term, g_dont, hitelarany_dont, inp['gepkap'], inp['gep_elettartam'], inp['gepar'],
#                       inp['futamido'], inp['hitel_kamatlab'], meglevo_gep0, selejtezes_inp, amort_inp, hiteltorl_inp,
#                       kamatfiz_inp)
# print(r2)


def merleg(eszkoz0, hitel0, beruh, amort, hitelfelv, hiteltorl):
    n = len(beruh)
    eszkoz_res = np.zeros(n)
    hitel_res = np.zeros(n)
    for i in range(n):
        if i == 0:
            eszkoz_res[i] = eszkoz0 + beruh[i] - amort[i]
            hitel_res[i] = hitel0 + hitelfelv[i] - hiteltorl[i]
        else:
            eszkoz_res[i] = eszkoz_res[i - 1] + beruh[i] - amort[i]
            hitel_res[i] = hitel_res[i - 1] + hitelfelv[i] - hiteltorl[i]
    return np.array([eszkoz_res, hitel_res])


def kuszob_arbevetel(valt_kts, fix_kts, kamatfiz, beruh, amort, hitelfelv, hiteltorl, adokulcs):
    arb = ((1 - adokulcs) * (valt_kts + fix_kts + kamatfiz) - amort * adokulcs + beruh - hitelfelv + hiteltorl) \
          / (1 - adokulcs)
    need_change = arb - valt_kts - fix_kts - amort - kamatfiz < 0
    indices = np.where(need_change)[0]
    arb[indices] = valt_kts[indices] + fix_kts[indices] + kamatfiz[indices] + beruh[indices] \
                   - hitelfelv[indices] + hiteltorl[indices]
    return arb


def get_valt(trend, szoras):
    valt = np.empty(3)
    valt[1] = trend
    valt[0] = trend + szoras
    valt[2] = trend - szoras
    return valt


def trinomfa(a0, valt, p, kuszob):
    n = len(kuszob)
    nn = 3 ** n
    res = [np.zeros((nn, n + 1)) for _ in range(3)]
    # 0: trinomiális fa értékek
    # 1: trinomiális fa valószínűségek
    # 2: csődbemenetel (0 vagy 1)
    res[0][(nn - 1) // 2, 0] = a0
    res[1][(nn - 1) // 2, 0] = 1
    elozohelyek = (nn - 1) // 2 * np.ones(1, dtype=int)
    for j in range(n):
        helyek = np.zeros(3 ** (j + 1), dtype=int)
        for i in range(3 ** j):
            nnn = nn // 3 ** (j + 1)
            helyek[3 * i] = elozohelyek[i] - nnn
            helyek[3 * i + 1] = elozohelyek[i]
            helyek[3 * i + 2] = elozohelyek[i] + nnn
            for k in range(3):
                res[0][helyek[3 * i + k], j + 1] = res[0][elozohelyek[i], j] * (1 + valt[k])
                res[1][helyek[3 * i + k], j + 1] = res[1][elozohelyek[i], j] * p[k]
                if res[2][elozohelyek[i], j] == 1:
                    res[2][helyek[3 * i + k], j + 1] = 1
                else:
                    if res[0][helyek[3 * i + k], j + 1] < kuszob[j]:
                        res[2][helyek[3 * i + k], j + 1] = 1
        elozohelyek = np.zeros(3 ** (j + 1), dtype=int)
        elozohelyek = helyek[:]
    return res


# a0=18
# valt=np.array([0.1, 0, -0.1])
# p=np.array([0.25, 0.5, 0.25])
# kuszob=17.99*np.ones(3)
# print(trinomfa(a0, valt, p, kuszob)[2])


def esetkiertekeles(a0, valt, p, kuszob):
    n = len(kuszob)
    res = np.zeros((2, n))
    # 0: csődvalószínűség az adott évig
    # 1: várható A az adott időszakban, feltéve, hogy a vállalat akkor még életben van
    trinomfa_res = trinomfa(a0, valt, p, kuszob)
    trinomfaertek = trinomfa_res[0]
    trinomfaval = trinomfa_res[1]
    trinomfacsod = trinomfa_res[2]
    for j in range(n):
        res[0, j] = min(np.matmul(trinomfaval[:, j + 1], np.transpose(trinomfacsod[:, j + 1])), 1)  # valamiért néha slightly nagyobb, mint 1
        if res[0, j] < 1:
            res[1, j] = np.sum(trinomfaertek[:, j + 1] * trinomfaval[:, j + 1] * (1 - trinomfacsod[:, j + 1])) / (
                        1 - res[0, j])
    return res


def calc_kuszob_a(term, term_others, kuszob_arb, b):
    res = kuszob_arb / term + b * (term + term_others)
    return res


def calc_ar(term, term_others, a, b):
    res = a - b * (term + term_others)
    return res


def calc_fcff_ts(term, term_others, inp, a_elozo, kamatfiz, beruh, amort, hitelfelv, hiteltorl):
    n = len(term)
    valt_kts = np.multiply(inp['valt_kts_c'], term)
    fix_kts = inp['fix_kts_f'] * np.ones(n)
    kuszob_arb = kuszob_arbevetel(valt_kts, fix_kts, kamatfiz, beruh, amort, hitelfelv, hiteltorl, inp['adokulcs'])
    kuszob_a = calc_kuszob_a(term, term_others, kuszob_arb, inp['b'])
    esetkiert = esetkiertekeles(a_elozo, get_valt(inp['trend'], inp['szoras']), inp['p'], kuszob_a)
    csodval = esetkiert[0, :]
    survival_val = 1 - csodval
    varhato_a = esetkiert[1, :]
    varhato_ar = calc_ar(term, term_others, varhato_a, inp['b'])
    # eredmény
    arb = varhato_ar * term
    ebitda = arb - valt_kts - fix_kts
    ebit = ebitda - amort
    ebt = ebit - kamatfiz
    ado = np.maximum(ebt * inp['adokulcs'], np.zeros(n))
    adozott_eredmeny = ebt - ado
    # cf
    uzemi_eredmeny_adoja = np.maximum(ebit * inp['adokulcs'], np.zeros(n))
    noplat = ebit - uzemi_eredmeny_adoja
    ocf = noplat + amort  # brutto cf = operativ cf
    fcff = ocf - beruh
    adopajzs = uzemi_eredmeny_adoja - ado
    return [fcff, adopajzs, survival_val]


# term_others = np.zeros(inp['tervezes_idoszakok'])


# kamatfiz = r2[6]
# beruh = r2[2]
# amort = r2[3]
# hitelfelv = r2[4]
# hiteltorl = r2[5]
# r3 = calc_fcff_ts(term, term_others, inp, inp['a0'], kamatfiz, beruh, amort, hitelfelv, hiteltorl)
# print(r3)


def calc_internal_value(ocf0, beruh0, ts0, fcff, ts, surv_val, g, elvart_hozam):
    fcff0 = ocf0 - beruh0
    n = len(fcff)
    res = fcff0 + ts0
    for i in range(n):
        res += (fcff[i] + ts[i]) * surv_val[i] / (1 + elvart_hozam) ** (i + 1)
    #     if i == n - 1:
    #         last_element = (fcff[i] + ts[i]) * surv_val[i] / (1 + elvart_hozam) ** (i + 1)
    # survival_discount = surv_val[-1] ** (1 / n)
    # discounted_g = (1 + g) * survival_discount - 1
    # r_minus_g = elvart_hozam - discounted_g  # max?
    # discounted_terminal_value = last_element * (1 + discounted_g) / r_minus_g
    # res += discounted_terminal_value
    return res


def calc_added_value(dont, dont_others, inp, hanyadik):
    term_dont = dont[0]
    g_dont = dont[1]
    hitelarany_dont = dont[2]
    term = np.squeeze(get_term_vector(term_dont, g_dont, inp['tervezes_idoszakok']))
    if not np.any(dont_others):  # if empty
        term_others = np.zeros(len(term))
    else:
        term_dont_others = dont_others[:,0]
        g_dont_others = dont_others[:,1]
        term_others = np.sum(get_term_vector(term_dont_others, g_dont_others, inp['tervezes_idoszakok']), axis=0)
    if inp['elsofutas']:
        hist = create_history(inp['gep_elettartam'], term_dont, hitelarany_dont)
        term_hist = hist[0]
        hitelarany_hist = hist[1]
        ocf0_ts0 = create_init_ocf_ts(term_dont, hitelarany_dont, inp['a0'], inp['b'], inp['valt_kts_c'],
                                      inp['fix_kts_f'], inp['adokulcs'], inp['hitel_kamatlab'], inp['gepar'],
                                      inp['gepkap'])
        inp['ocf0'][hanyadik] = ocf0_ts0[0]
        inp['ts0'][hanyadik] = ocf0_ts0[1]
        init_inp = calc_from_init_history(term_hist, hitelarany_hist, inp['gepkap'], inp['gep_elettartam'],
                                          inp['gepar'], inp['futamido'], inp['hitel_kamatlab'], inp['tervezes_idoszakok'])
        inp['meglevo_gep0'][hanyadik] = init_inp[0]
        inp['selejtezes_inp'][hanyadik,:] = init_inp[1]
        inp['amort_inp'][hanyadik,:] = init_inp[2]
        inp['hiteltorl_inp'][hanyadik,:] = init_inp[3]
        inp['kamatfiz_inp'][hanyadik,:] = init_inp[4]
        inp['eszkoz0'][hanyadik] = init_inp[5]
        inp['hitel0'][hanyadik] = init_inp[6]
        inp['amort0'][hanyadik] = 0
        inp['hiteltorl0'][hanyadik] = 0

    res1 = calc_beruh_hitel(term, g_dont, hitelarany_dont, inp['gepkap'], inp['gep_elettartam'], inp['gepar'],
                            inp['futamido'], inp['hitel_kamatlab'],
                            inp['meglevo_gep0'][hanyadik], inp['selejtezes_inp'][hanyadik,:],
                            inp['amort_inp'][hanyadik,:], inp['hiteltorl_inp'][hanyadik,:], inp['kamatfiz_inp'][hanyadik,:])
    beruh0 = res1[0]
    hitelfelv0 = res1[1]
    ic = merleg(inp['eszkoz0'][hanyadik], inp['hitel0'][hanyadik], [beruh0], [inp['amort0'][hanyadik]], [hitelfelv0],
                [inp['hiteltorl0'][hanyadik]])[0]
    beruh = res1[2]
    amort = res1[3]
    hitelfelv = res1[4]
    hiteltorl = res1[5]
    kamatfiz = res1[6]
    res2 = calc_fcff_ts(term, term_others, inp, inp['a0'], kamatfiz, beruh, amort, hitelfelv, hiteltorl)
    fcff = res2[0]
    adopajzs = res2[1]
    surv_val = res2[2]
    internal_value = calc_internal_value(inp['ocf0'][hanyadik], beruh0, inp['ts0'][hanyadik], fcff, adopajzs, surv_val, g_dont, inp['elvart_hozam'])
    added_value = internal_value - ic
    return -added_value  # mínusz az optimalizálás miatt


# dont_others = np.delete(np.array([[1000, 0, 0]]), 0, 0)
# added_value = calc_added_value(dont, dont_others, inp, 0)
# print(added_value)
#
# start_time = time.time()
# dont1 = optimize.direct(calc_added_value, ((1, inp['a0']/inp['b']), (0,1), (0,1)), args=([0,0,0], inp, 0), maxfun=500)
# print('direct: ' + str(time.time()-start_time))
# print(dont1)
# #
# start_time = time.time()
# dont2 = optimize.brute(calc_added_value, ((1, inp['a0']/inp['b']), (0,1), (0,1)), args=([0,0,0], inp, 0), Ns=8)
# print('brute: ' + str(time.time()-start_time))
# print(dont2)
#
# added_value = calc_added_value(dont2, dont_others, inp, 0)
# print(added_value)
#
# start_time = time.time()
# dont3 = optimize.differential_evolution(calc_added_value, ((1, inp['a0']/inp['b']), (0,1), (0,1)), args=([0,0,0], inp, 0))
# print('diff_evol: ' + str(time.time()-start_time))
# print(dont3)
#
# start_time = time.time()
# dont4 = optimize.shgo(calc_added_value, ((1, inp['a0']/inp['b']), (0,1), (0,1)), args=([0,0,0], inp, 0))
# print('shgo: ' + str(time.time()-start_time))
# print(dont4)

# start_time = time.time()
# dont5 = optimize.dual_annealing(calc_added_value, ((1, inp['a0']/inp['b']), (0,1), (0,1)), args=([0,0,0], inp, 0), maxiter=200)
# print('dual_ann: ' + str(time.time()-start_time))
# print(dont5)

# start_time = time.time()
# dont6 = optimize.basinhopping(calc_added_value, [1000, 0, 0.5], minimizer_kwargs={'args':([0,0,0], inp, 0)})
# print('basinhopping: ' + str(time.time()-start_time))
# print(dont6)

# added_value = calc_added_value(dont, [0,0,0], inp, inp['a0'],
#                                meglevo_gep0, selejtezes_inp, amort_inp, hiteltorl_inp, kamatfiz_inp,
#                                eszkoz0, hitel0, 0, 0, ocf0, ts0)
# print(added_value)

# print(optimize.basinhopping(calc_added_value, [1000, 0, 0.5], minimizer_kwargs={'args': ([0,0,0], inp, inp['a0'], meglevo_gep0, selejtezes_inp, amort_inp, hiteltorl_inp, kamatfiz_inp, eszkoz0, hitel0, 0, 0, ocf0, ts0)}))


def optim(dont_others, inp, hanyadik):
    inp['szamlalo'] += 1
    if inp['szamlalo'] == 27:
        inp['szamlalo'] += 0
    return optimize.direct(calc_added_value, ((1, inp['a0']/inp['b']), (0,1), (0,1)), args=(dont_others, inp, hanyadik),
                           maxfun=500).x


def reakcio_vektor(dont_all, inp):
    k = len(dont_all) // 3
    dont_all = np.reshape(dont_all, (k, 3))
    sh = dont_all.shape
    res = np.empty(sh)
    for i in range(sh[0]):
        dont_others = np.delete(dont_all, i, axis=0)
        res[i,:] = optim(dont_others, inp, i)
    return np.squeeze(np.reshape(dont_all-res, (1, k*3)))


def nash_egyensuly(inp):
    orig_shape = inp['x0'].shape
    res = optimize.fsolve(reakcio_vektor, inp['x0'], args=(inp, ), maxfev=400)
    return np.reshape(res, orig_shape)


# x0 = np.array([dont])
# print(nash_egyensuly(inp))
