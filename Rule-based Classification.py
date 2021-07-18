#############################################################
# ÖDEV 3: Kural Tabanlı Sınıflandırma ile
# Potansiyel Müşteri Getirisi Hesaplama
#############################################################
import numpy as np
import pandas as pd

df = pd.read_csv("datasets/persona.csv")


# Değişkenler
# persona.csv

# PRICE – Müşterinin harcama tutarı
# SOURCE – Müşterinin bağlandığı cihaz türü
# SEX – Müşterinin cinsiyeti
# COUNTRY – Müşterinin ülkesi
# AGE – Müşterinin yaşı




# Uygulama Öncesi Veri Seti

# customers_level_based       PRICE           SEGMENT
# BRA_ANDROID_FEMALE_0_18     1139.800000     A
# BRA_ANDROID_FEMALE_19_23    1070.600000     A
# BRA_ANDROID_FEMALE_24_30    508.142857      A
# BRA_ANDROID_FEMALE_31_40    233.166667      C
# BRA_ANDROID_FEMALE_41_66    236.666667      C


# Hedeflenen çıktı

# PRICE       SOURCE          SEX     COUNTRY     AGE
# 39          android         male    bra         17
# 39          android         male    bra         17
# 49          android         male    bra         17
# 29          android         male    tur         17
# 49          android         male    tur         17





#  GÖREV 1:
# Aşağıdaki soruları yanıtlayınız.
# ▪ Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("datasets/persona.csv")
df.head(10)
df.tail(10)
df.info()
df.shape
df.ndim
df.dtypes
# ▪ Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()
# ▪ Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()
# ▪ Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts().sum
# ▪ Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df.groupby("COUNTRY").agg({"PRICE": "count"})
df.groupby("AGE").agg({"PRICE": "count"})
# ▪ Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# ▪ Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
df.groupby("SOURCE").agg({"PRICE": "count"})
# ▪ Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE": "mean"})
# ▪ Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE": "mean"})
# ▪ Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["SOURCE","COUNTRY"]).agg({"PRICE": "mean"})




# GÖREV 2:
# COUNTRY, SOURCE, SEX, AGE kırılımında
# toplam kazançlar nedir?

# ÖRNEK ÇIKTI
#     COUNTRY     SOURCE      SEX        AGE   PRICE
# 0   bra         android     female     15    1355
# 1                                      16    1294
# 2                                      17    642
# 3                                      18    1387
# 4                                      19    1021
df.groupby(["SOURCE","COUNTRY","SEX","AGE"]).agg({"PRICE": "sum"})



# GÖREV 3:
# Çıktıyı PRICE’a göre sıralayınız.
# ▪ Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan
# olacak şekilde PRICE’a göre uygulayınız.
# ▪ Çıktıyı agg_df olarak kaydediniz.

# ÖRNEK ÇIKTI
# COUNTRY     SOURCE      SEX     AGE     PRICE
# usa         android     male    15      3917
# bra         android     male    19      2606
# usa         ios         male    15      2496
#             android     female  20      2190
# deu         ios         female  16      2169

agg_df = df.groupby(["SOURCE","COUNTRY","SEX","AGE"]).agg({"PRICE": "sum"}).sort_values(by= "PRICE",ascending=False)


# GÖREV 4:
# Index’te yer alan isimleri değişken ismine çeviriniz.

# ▪ Üçüncü sorunun çıktısında yer alan price dışındaki tüm değişkenler index isimleridir.
# ▪ Bu isimleri değişken isimlerine çeviriniz.
#iPUCU: reset_index()

agg_df = agg_df.reset_index()

# GÖREV 5:

# age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# ▪ Age sayısal değişkenini kategorik değişkene çeviriniz.
# ▪ Aralıkları ikna edici şekilde oluşturunuz.
# ▪ Örneğin: '0_19', '20_24', '24_31', '31_41', '41_70'

#ÖRNEK ÇIKTI:
# COUNTRY     SOURCE      SEX     AGE     PRICE   AGE_CAT
# usa         android     male    15      3917    0_18
# bra         android     male    19      2606    0_18
# usa         ios         male    15      2496    0_18
# usa         android     female  20      2190    19_23
# deu         ios         female  16      2169    0_18

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0,18,24,31,41,70], labels=["0_18" ,"19_24" , "25_31" , "32_41" , "42_70"])




# GÖREV 6:
#  Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# ▪ Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# ▪ Yeni eklenecek değişkenin adı: customers_level_based
# ▪ Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based
# değişkenini oluşturmanız gerekmektedir.


# Bu tabloda bulunan gözlemler bir araya gelecek

# COUNTRY     SOURCE      SEX     AGE     PRICE   AGE_CAT
# usa         android     male    15      3917    0_18
# bra         android     male    19      2606    0_18
# usa         ios         male    15      2496    0_18
# usa         android     female  20      2190    19_23
# deu         ios         female  16      2169    0_18


# Elde edilmesi gereken çıktı

# customers_level_based       PRICE
# USA_ANDROID_MALE_0_18       3917
# BRA_ANDROID_MALE_0_18       2606
# USA_IOS_MALE_0_18           2496
# USA_ANDROID_FEMALE_19_23    2190
# DEU_IOS_FEMALE_0_18         2169



# Dikkat! List comprehension ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18. Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.



colon = [col for col in agg_df.columns if col not in ["AGE","PRICE"]]
agg_df["customers_level_based"] = ["_".join(i).upper() for i in agg_df[colon].values]

# def join_columns(dataframe, columns, upper=True):
#     if upper:
#         return ["_".join(value).upper() for value in dataframe[columns].values]
#     else:
#         return ["_".join(value) for value in dataframe[columns].values]



agg_df_2 = agg_df[["customers_level_based","PRICE"]]
agg_df_2 = agg_df_2.groupby("customers_level_based").agg({"PRICE":"mean"})




# GÖREV 7:
# Yeni müşterileri (personaları) segmentlere ayırınız.
# ▪ Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# ▪ Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# ▪ Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
# ▪ C segmentini analiz ediniz (Veri setinden sadece C segmentini çekip analiz ediniz).

# İPUCU:
# pd.qcut(agg_df["PRICE"], 4, labels = ["D", "C", "B", "A"])
agg_df_2["SEGMENT"] = pd.qcut(agg_df_2["PRICE"], 4, labels = ["D", "C", "B", "A"])
agg_df_2 = agg_df_2.reset_index()
agg_df_2.groupby("SEGMENT").nunique()
agg_df_2.groupby("SEGMENT").agg({"PRICE":["mean","max","min","sum"]})


agg_df_2[agg_df_2["SEGMENT"] == "C"].describe().T



# GÖREV 8:

# Yeni gelen müşterileri segmentlerine göre sınıflandırınız ve
# ne kadar gelir getirebileceğini tahmin ediniz.
# ▪ 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve
# ortalama ne kadar gelir kazandırması beklenir?
# ▪ 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne
# kadar gelir kazandırması beklenir?

# İPUCU :
# new_user = "TUR_ANDROID_FEMALE_31_40"
# agg_df[agg_df["customer_level_based"] == new_user]


new_user = "ANDROID_TUR_FEMALE_32_41"
agg_df_2[agg_df_2["customers_level_based"] == new_user]

new_user_2 = "IOS_FRA_FEMALE_32_41"
agg_df_2[agg_df_2["customers_level_based"] == new_user_2]
