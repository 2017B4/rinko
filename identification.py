#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
HMM勉強用モジュール~評価問題によるHMM識別ver.~

requirements
* python 3.6
* numpy 1.12.1 
* hmmlearn 0.2.0
* enshu_hmm <-演習用ファイルの関数を再利用
"""
from hmmlearn import hmm
import numpy as np
import enshu_hmm

def def_sunny_param():
    u"""
    晴れが多いHMMのパラメータを設定

    隠れ状態、出力記号及び各パラメータの設定

    設定する内容
    状態　　：'雨'、'晴れ'の2つ
    出力記号：'散歩'、'買い物'、'掃除'の3つ

    ______________________________________________
    返り値       (type)  :content
    ______________________________________________
    |states       (tuple):隠れ状態
    |observations (tuple):出力記号
    |s            (dic)  :startprob_（初期状態確率）
    |t            (dic)  :transmat_(状態遷移確率)
    |e            (dic)  :emissionprob_(出力確率)
    """

    states = ('雨', '晴れ') # 状態の定義
    observations = ('散歩','買い物','掃除') # ボブの行動の定義

    s = {'雨':0.2, '晴れ':0.8} # 初期状態確率

    t = { # 各状態における状態遷移確率
        '雨': {'雨':0.2, '晴れ':0.8},
        '晴れ': {'雨':0.2, '晴れ':0.8},
    }

    e = { # 各状態における出力確率
        '雨': {'散歩':0.1, '買い物':0.4, '掃除':0.5},
        '晴れ': {'散歩':0.6, '買い物':0.3, '掃除':0.1},
    }
    #状態、出力記号、初期状態確率、状態遷移確率、出力確率の順に値を返す
    return states,observations,s,t,e

def def_rainy_param():
    u"""
    晴れが多いHMMのパラメータを設定

    隠れ状態、出力記号及び各パラメータの設定

    設定する内容
    状態　　：'雨'、'晴れ'の2つ
    出力記号：'散歩'、'買い物'、'掃除'の3つ

    ______________________________________________
    返り値       (type)  :content
    ______________________________________________
    |states       (tuple):隠れ状態
    |observations (tuple):出力記号
    |s            (dic)  :startprob_（初期状態確率）
    |t            (dic)  :transmat_(状態遷移確率)
    |e            (dic)  :emissionprob_(出力確率)
    """

    states = ('雨', '晴れ') # 状態の定義
    observations = ('散歩','買い物','掃除') # ボブの行動の定義

    s = {'雨':0.8, '晴れ':0.2} # 初期状態確率

    t = { # 各状態における状態遷移確率
        '雨': {'雨':0.8, '晴れ':0.2},
        '晴れ': {'雨':0.8, '晴れ':0.2},
    }

    e = { # 各状態における出力確率
        '雨': {'散歩':0.1, '買い物':0.4, '掃除':0.5},
        '晴れ': {'散歩':0.6, '買い物':0.3, '掃除':0.1},
    }
    #状態、出力記号、初期状態確率、状態遷移確率、出力確率の順に値を返す
    return states,observations,s,t,e
    
if __name__ == "__main__":
    states,observations,s,t,e  =def_sunny_param()
    sunny_model = enshu_hmm.make_hmm(states,observations,s,t,e)
    print("晴れが多い地域(sunny_model)のパラメータを表示します")
    print("初期状態確率:{},\n状態遷移確率:{},\n出力確率:{}\n".format(s,t,e))

    states,observations,s,t,e = def_rainy_param()
    rainy_model = enshu_hmm.make_hmm(states,observations,s,t,e)
    print("雨が多い地域(rainy_model)のパラメータを表示します")
    print("初期状態確率:{},\n状態遷移確率:{},\n出力確率:{}\n".format(s,t,e))

    sun_X = sunny_model.sample(10)[0]
    rain_X = rainy_model.sample(10)[0]

    sun_bob = [observations[int(x)] for x in sun_X]    
    rain_bob = [observations[int(x)] for x in rain_X]    

    print("観測系列を選択してください(0 or 1 or 2)\n")
    print("0. プログラム終了")
    print("1. 晴れが多い地域:{}".format(sun_bob))
    print("2. 雨　が多い地域:{}\n".format(rain_bob))

    while(1):
        which = input(">> ")
        if which == '0':
            break
        elif which == "1":
            print("観測系列1.について評価。尤度を比較します。")
            sun_score = np.exp(sunny_model.score(sun_X))
            rain_score = np.exp(rainy_model.score(sun_X))
            if sun_score > rain_score:
               ans = "sunny_model"
            else :
               ans = "rainy_model"
            print("sunny_modelの尤度S:{:10f}".format(sun_score))
            print("rainy_modelの尤度R:{:10f}".format(rain_score))
            print("S > R == {}\n".format(sun_score > rain_score))
            print("結果: 1.の観測系列を出力したモデルは{}と判定しました。".format(ans))
        elif which == "2":
            print("観測系列2.について評価。尤度を比較します。")
            sun_score = np.exp(sunny_model.score(rain_X))
            rain_score = np.exp(rainy_model.score(rain_X))
            if sun_score > rain_score:
               ans = "sunny_model"
            else :
               ans = "rainy_model"
            print("sunny_modelの尤度S:{:10f}".format(sun_score))
            print("rainy_modelの尤度R:{:10f}".format(rain_score))
            print("R > S == {}\n".format(rain_score > sun_score))
            print("結果: 2.の観測系列を出力したモデルは{}と判定しました。".format(ans))

        else:
            print("'1'か'2'で答えてください.終了するときは'0'を入力してください.")
    
