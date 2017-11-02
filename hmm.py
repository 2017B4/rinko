#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
HMM勉強用モジュール

requirements
* python 3.6
* numpy 1.12.1 
* hmmlearn 0.2.0
"""
from hmmlearn import hmm
import numpy as np

def def_param():
    u"""
    HMMのパラメータを設定

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

    s = {'雨':0.6, '晴れ':0.4} # 初期状態確率

    t = { # 各状態における状態遷移確率
        '雨': {'雨':0.7, '晴れ':0.3},
        '晴れ': {'雨':0.4, '晴れ':0.6},
    }

    e = { # 各状態における出力確率
        '雨': {'散歩':0.1, '買い物':0.4, '掃除':0.5},
        '晴れ': {'散歩':0.6, '買い物':0.3, '掃除':0.1},
    }
    #状態、出力記号、初期状態確率、状態遷移確率、出力確率の順に値を返す
    return states,observations,s,t,e

def make_hmm(states,observations,s,t,e):
    u"""
    HMMを生成する

    ______________________________________________
    引数          (type) :content
    ______________________________________________
    |states       (tuple):隠れ状態
    |observations (tuple):出力記号
    |s            (dic)  :startprob_（初期状態確率）
    |t            (dic)  :transmat_(状態遷移確率)
    |e            (dic)  :emissionprob_(出力確率)
    ______________________________________________

    ______________________________________________
    返り値        (type) :content
    ______________________________________________
    |model        (class):生成したHMMのインスタンス
    """
    model = hmm.MultinomialHMM(n_components=2) # Ergodicの離散型隠れマルコフモデル,状態数：２


    # 初期状態確率
    # 1 * 状態数
    start = np.array([s['雨'],s['晴れ']])
    # 状態遷移確率
    # 状態数 * 状態数
    trans = np.array([[t['雨']['雨'],t['雨']['晴れ']],
                      [t['晴れ']['雨'],t['晴れ']['晴れ']]
                     ])
    # 出力確率
    # 状態数 * 出力記号数
    emiss = np.array([[e['雨']['散歩'],e['雨']['買い物'],e['雨']['掃除']],
                      [e['晴れ']['散歩'],e['晴れ']['買い物'],e['晴れ']['掃除']]
                     ])

    # モデルにパラメータを設定
    model.startprob_ = start # 初期状態確率
    model.transmat_ = trans # 状態遷移確率
    model.emissionprob_ = emiss # 出力確率

    return model # 生成したモデルを返す

def make_sample(model,states,observations):
    u"""
    HMMからサンプルデータを出力する

    HMMを動かして,ある状態遷移が行われた時の
    観測系列と状態遷移系列を得る
    
    ______________________________________________
    引数          (type) :content
    ______________________________________________
    |model        (class):HMMのインスタンス
    |states       (tuple):隠れ状態
    |observations (tuple):出力記号
    ______________________________________________

    ______________________________________________
    返り値        (type) :content
    ______________________________________________
    |X1        (np.array):modelから出力された観測系列
    |Z1        (np.array):X1が出力された時の状態遷移系列
    """
    # サンプルデータの出力
    # X = 観測系列、Z = 観測系列がXの時の状態系列
    X1,Z1 = model.sample(10) 

    print("サンプルデータを出力します。")
    for x in range(len(X1)):
        print("{0}日目の天気は'{1}'で、ボブは'{2}'をしていました。".format(x+1,states[Z1[x]], observations[X1[x][0]]))

    print("この時のボブの行動の尤度：{0:10f}\n".format(np.exp(model.score(X1))))
    return X1,Z1

def Predict(model, X1,Z1):
    u"""
    復号問題を解いて最尤状態遷移系列を求める.
    
    ビタビアルゴリズムを用いて,観測系列を元に
    最尤状態遷移系列を推定する.
    ______________________________________________
    引数          (type) :content
    ______________________________________________
    |model        (class):HMMのインスタンス
    |X1        (np.array):make_sampleで出力された観測系列
    |Z1        (np.array):X1を出力した時の状態系列
    ______________________________________________

    ______________________________________________
    返り値       (type) :content
    ______________________________________________
    |なし
    """
    Pre_Z1 = model.predict(X1) # model.predictメソッドに観測系列X1を渡して状態系列を最尤推定

    print("復号結果を表示します")
    ans_cnt = 0
    for x in range(len(X1)):
        print("{0}日目,ボブは{1}をしており、天気は'{2}'と予測しました。".format(x+1, observations[X1[x][0]],states[Pre_Z1[x]]))
        if Z1[x] == Pre_Z1[x]: # 元の状態系列と、最尤推定した状態系列の一致数を求める
            ans_cnt = ans_cnt+1
    
    print("予測した天気の正解数は{0}個中、{1}個でした。\n".format(len(Z1),ans_cnt))
    print(type(X1),type(Z1))

def Estimate(model,X1,Z1):
    u"""
    HMMのパラメータの推定を行う。

    バウムウェルチアルゴリズムを用いて,未知のHMMから
    出力された観測系列を元に,HMMの各パラメータを推定する.
    ______________________________________________
    引数          (type) :content
    ______________________________________________
    |model        (class):HMMのインスタンス
    |X1        (np.array):make_sampleで出力された観測系列
    |Z1        (np.array):X1を出力した時の状態系列
    ______________________________________________

    ______________________________________________
    返り値       (type) :content
    ______________________________________________
    |なし
    """
    # HMMのインスタンスを生成
    # n_iter は推定を行う演算のイテレーションの回数
    remodel = hmm.MultinomialHMM(n_components=2,n_iter=10000)
    # fitメソッドに観測系列を渡してパラメータを推定
    # この時、X1はmodelからの出力だが,"未知"のモデルから出力されたものと仮定して学習する
    remodel.fit(X1)
    # 学習済みのremodelからサンプルデータを出力
    X2,Z2 = remodel.sample(10)

    print("パラメータ推定を行ったモデルからサンプルを出力します。")
    for x in range(len(X1)):
        print("{0}日目の天気は'{1}'で、ボブは'{2}'をしていました。".format(x+1,states[Z2[x]], observations[X2[x][0]]))
    
    #要検証部分なのでスルーしてください
    #print("元のモデルとパラメータ推定したモデルの出力結果を比較します。")
    #print("観測系列の一致数は{0}個中{1}個でした。".format(len(X1),len([x for x in range(len(X1)) if X1[x] == X2[x]])))
    #print("状態推移の一致数は{0}個中{1}個でした。".format(len(Z1),len([x for x in range(len(Z1)) if Z1[x] == Z2[x]])))
if __name__ == "__main__":
    # hmmのパラメータを取得
    states,observations,s,t,e = def_param()
    # HMMのインスタンスを生成
    model = make_hmm(states,observations,s,t,e)
    X1,Z1 = make_sample(model,states,observations)
    Predict(model,X1,Z1)
    Estimate(model,X1,Z1)
