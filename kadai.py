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

#サンプル出力の数
SAMPLE = 10

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

    print("状態集合：{}".format(states))
    print("出力記号集合:{}\n".format(observations))
    ####################################################################
    #############実習で求めたパラメータを入力する箇所です！#############
    s = {'雨':, '晴れ':} # 初期状態確率

    t = { # 各状態における状態遷移確率
        '雨': {'雨':, '晴れ':},
        '晴れ': {'雨':, '晴れ':},
    }

    e = { # 各状態における出力確率
        '雨': {'散歩':, '買い物':, '掃除':},
        '晴れ': {'散歩':, '買い物':, '掃除':},
    }
    ####################################################################
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
    X1,Z1 = model.sample(SAMPLE) 

    print("modelからサンプルデータを出力します。")
    for x in range(len(X1)):
        print("{0}日目の天気は'{1}'で、ボブは'{2}'をしていました。".format(x+1,states[Z1[x]], observations[X1[x][0]]))

    print("この時のボブの行動の尤度：{0:10f}%\n".format(np.exp(model.score(X1))*100))
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

    print("サンプルの観測系列からmodelにおける最尤状態遷移系列を復号します")
    print("{:*^10}".format("復号結果"))
    ans_cnt = 0
    for x in range(len(X1)):
        print("{0}日目,ボブは{1}をしており、天気は'{2}'と予測しました。".format(x+1, observations[X1[x][0]],states[Pre_Z1[x]]))
        if Z1[x] == Pre_Z1[x]: # 元の状態系列と、最尤推定した状態系列の一致数を求める
            ans_cnt = ans_cnt+1
    
    print("予測した天気の正解数は{0}個中、{1}個でした。\n".format(len(Z1),ans_cnt))

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
    remodel = hmm.MultinomialHMM(n_components=2,n_iter=10)
       
    # fitメソッドに観測系列を渡してパラメータを推定

    # modelから10000日分の出力を観測系列として学習する
    #print("modelからの出力10000日分の観測系列をremodelで学習します")
    print("統計情報の観測系列からremodelを学習します")
    remodel.fit(model.sample(10000)[0])
    #remodelに入れる観測系列が決まったら更新すること

    # 学習済みのremodelに関して
    #サンプルデータの観測系列から最尤状態遷移を復号する
    Pre_Z1=remodel.predict(X1)

    ans_cnt=0
    print("modelのサンプルからremodelにおける最尤状態遷移系列を復号します")
    print("{:*^10}".format("復号結果"))
    for x in range(len(X1)):
        print("{0}日目,ボブは{1}をしており、天気は'{2}'と予測しました。".format(x+1, observations[X1[x][0]],states[Pre_Z1[x]]))
        if Z1[x] == Pre_Z1[x]: # 元の状態系列と、最尤推定した状態系列の一致数を求める
            ans_cnt = ans_cnt+1
    
    print("予測した天気の正解数は{0}個中、{1}個でした。\n".format(len(Z1),ans_cnt))

    return remodel    
def show_param(remodel, s,t,e):
    u"""
    HMMのパラメータを表示する。

    自分でパラメータを定義したmodelのパラメータを表示した後、
    Estimateによって推定したモデルのパラメータを表示する。
    ______________________________________________
    引数          (type) :content
    ______________________________________________
    |remodel      (class):パラメータ推定したHMMのインスタンス
    |s            (dic)  :startprob_（初期状態確率）
    |t            (dic)  :transmat_(状態遷移確率)
    |e            (dic)  :emissionprob_(出力確率)
    ______________________________________________

    ______________________________________________
    返り値       (type) :content
    ______________________________________________
    |なし
    """

    print("元のモデルのパラメータを表示します")
    print("初期状態確率:{},\n状態遷移確率:{},\n出力確率:{}\n".format(s,t,e))
    
    # remodelの初期状態確率、状態遷移確率、出力確率を表示する
    # formatは桁数の調整
    s['雨'] = format(remodel.startprob_[0], '.2f')
    s['晴れ'] = format(remodel.startprob_[1], '.2f')
    
    t['雨']['雨'] = format(remodel.transmat_[0][0], '.2f')
    t['雨']['晴れ'] = format(remodel.transmat_[0][1], '.2f')
    t['晴れ']['雨'] = format(remodel.transmat_[1][0], '.2f')
    t['晴れ']['晴れ']= format(remodel.transmat_[1][1], '.2f')

    e['雨']['散歩']=format(remodel.emissionprob_[0][0], '.2f') 
    e['雨']['買い物']=format(remodel.emissionprob_[0][1], '.2f')
    e['雨']['掃除']=format(remodel.emissionprob_[0][2], '.2f')
    e['晴れ']['散歩']=format(remodel.emissionprob_[1][0], '.2f') 
    e['晴れ']['買い物']=format(remodel.emissionprob_[1][1], '.2f')
    e['晴れ']['掃除']=format(remodel.emissionprob_[1][2], '.2f')
    
    print("推定したモデルのパラメータを表示します")
    print("初期状態確率:{},\n状態遷移確率:{},\n出力確率:{}".format(s,t,e))

if __name__ == "__main__":
    # hmmのパラメータを取得
    states,observations,s,t,e = def_param()
    # HMMのインスタンスを生成
    model = make_hmm(states,observations,s,t,e)
    # modelからサンプルデータを得る
    X1,Z1 = make_sample(model,states,observations)
    # modelとサンプルデータから復号問題を解く
    Predict(model,X1,Z1)
    # modelから得た観測系列を元にremodelのパラメータを推定する
    remodel = Estimate(model,X1,Z1)
    # model,remodelのパラメータを表示する
    show_param(remodel,s,t,e)
