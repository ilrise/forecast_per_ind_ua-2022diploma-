import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import  combinations
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_percentage_error,\
                            mean_squared_error


w=str(input("Введіть назву файлу:"))
file_path = f'{w}.xlsx'

df = pd.read_excel(file_path, sheet_name=0)
product = ['01 живi тварини','02 м’ясо та їстівні субпродукти','03 риба i ракоподібні','04 молоко та молочнi продукти яйця птиці; натуральний мед','05 інші продукти тваринного походження',
'06 живі дерева та інші рослини', '07 овочі', '08 їстівні плоди та горіхи', '09 кава, чай','10 зерновi культури','11 продукцiя борошномельно-круп’яної промисловості','12 насiння і плоди олійних рослин',
'13 шелак природний','14 рослинні матеріали для виготовлення','15 жири та олії тваринного або рослинного походження','16 продукти з м’яса, риби','17 цукор і кондитерські вироби з цукру',
'18 какао та продукти з нього','19 готові продукти із зерна','20 продукти переробки овочів','21 різні харчовi продукти','22 алкогольнi i безалкогольнi напої та оцет',
'23 залишки і вiдходи харчової промисловості','24 тютюн і промислові замінники тютюну','25 сiль; сірка; землі та каміння','26 руди, шлак і зола','27 палива мінеральні; нафта і продукти її перегонки',
'28 продукти неорганiчної хімії','29 органiчнi хiмiчнi сполуки','30 фармацевтична продукція','31 добрива','32 екстракти дубильні','33 ефiрнi олії','34 мило, поверхнево-активні органічні речовини',
'35 бiлковi речовини','36 порох і вибуховi речовини','37 фотографічні або кiнематографічні товари','38 різноманітна хімічна продукція','39 пластмаси, полімерні матеріали','40 каучук, гума',
'41 шкури','42 вироби із шкіри','43 натуральне та штучне хутро','44 деревина і вироби з деревини','45 корок та вироби з нього','46 вироби із соломи','47 маса з деревини','48 папiр та картон',
'49 друкована продукція','50 шовк','51 вовна','52 бавовна','53 іншi текстильнi волокна','54 нитки синтетичні або штучні','55 синтетичні або штучні штапельнi волокна',
'56 вата','57 килими','58 спецiальнi тканини','59 текстильнi матеріали','60 трикотажні полотна','61 одяг та додаткові речі до одягу, трикотажні','62 одяг та додаткові речі до одягу, текстильні',
'63 іншi готовi текстильні вироби','64 взуття','65 головнi убори','66 парасольки','67 обробленi пір’я та пух','68 вироби з каменю, гiпсу, цементу','69 керамiчнi вироби','70 скло та вироби із скла',
'71 перли природні або культивовані, дорогоцінне або напівдорогоцінне каміння','72 чорнi метали','73 вироби з чорних металів','74 мiдь i вироби з неї','75 нiкель i вироби з нього',
'76 алюмiнiй i вироби з нього','77 свинець і вироби з нього','78 цинк i вироби з нього','79 олово і вироби з нього','80 іншi недорогоцінні метали','81 інструменти, ножовi вироби',
'82 іншi вироби з недорогоцінних металiв','83 реактори ядерні, котли, машини','84 електричнi машини','85 залізничні локомотиви','86 засоби наземного транспорту, крiм залізничного',
'87 літальні апарати','88 судна','89 прилади та апарати оптичнi, фотографічні','90 годинники','91 музичні інструменти','92 меблi','93 іграшки','94 рiзнi готовi вироби','95 твори мистецтва'
]

# print(len(product))
# w=1
# for w in range(len(product)):
#     print(product[w])

g=0

while g==0:

        a=int(input("Ввести число від 0 до 95="))
        # назва групи товарів
        name_product=product[a-1]
        print(name_product)
        all=df[df['group']==a]
        q1=all['cnt_ex'].isnull().sum()
        q2=all['cnt_im'].isnull().sum()
        q3=all['prc_ex'].isnull().sum()
        q4=all['prc_im'].isnull().sum()
        q5=all['g_qnt'].isnull().sum()
        q6=all['g_prc'].isnull().sum()
        all_new = all.interpolate(method='polynomial', order=2)

        missing_values=(all['cnt_im'].isnull().sum()+all['cnt_ex'].isnull().sum()+all['prc_ex'].isnull().sum()+all['prc_im'].isnull().sum()+all['g_qnt'].isnull().sum()+all['g_prc'].isnull().sum())
        if q1<5 and q2<5 and q3<5 and q4<5 and q5<5 and q6<5:

            #заповнення пропусків датафрейму значеннями
            all_new=all.interpolate(method='polynomial', order=2)

            # #матриця кореляційна
            # cols = ['cnt_ex', 'cnt_im', 'prc_ex','prc_im', 'g_qnt', 'g_prc']
            #
            # corr = all_new[cols].corr().round(2)
            #
            # plt.figure(figsize=(7, 7))
            # plt.title('Correlation matrix')
            # sns.heatmap(corr, square=True, cmap='Spectral', annot=True)
            # plt.xticks(rotation=70)
            # plt.show()
            print('1 - індекси фізичного обсягу експорту Ласпейреса')
            print('2 - індекси фізичного обсягу імпорту Ласпейреса')
            print('3 - індекc цін експорту Пааше')
            print('4 - індекc цін імпорту Пааше')
            print('5 - індекс умов торгівлі кількісний')
            print('6 - індекс умов торгівлі ціновий')
            ans=int(input('Введіть номер індексу = '))

            if ans==1:
            #ARIMA MODEL індекси фізичного обсягу експорту Ласпейреса
                independent_cols =  ['cnt_im', 'prc_ex','prc_im', 'g_qnt', 'g_prc']              # прогнозування фізичного обсягу

                X_train, X_test, y_train, y_test = train_test_split(
                    all_new[independent_cols], all_new['cnt_ex'],
                    test_size=0.2, shuffle=False)
                min_1 = 99999999
                min_2=999999999
                #parameters p,q,r
                for p in range(5):
                    for q in range(5):
                            for r in range(5):
                                perm = combinations((p, q, r),3)
                                for i in list(perm):

                                    model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=i).fit()
                                    fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                                         alpha=0.05).summary_frame()
                                    mape = round(mean_absolute_percentage_error(y_test, fcast['mean']), 2)
                                    mse = round(mean_squared_error(y_test, fcast['mean']), 2)
                                    print(i,mape,mse)
                                    if mape<min_1 and mse<min_2:
                                        min_1,min_2,best_o=mape,mse,i
                                        continue
                                    else:
                                        continue
                print('Best ARIMA%s MAPE=%.3f MSE=%.3f' % (best_o, min_1,min_2))



                model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=best_o).fit()                     # вместо 1,3,1 - p,q,r - MAPE и MSE должны быть минимальны
                print(model_arima.summary())
                fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                  alpha=0.05).summary_frame()  # 95% conf
                print('MAPE test:', round(mean_absolute_percentage_error(y_test, fcast['mean']), 2))   #ошибка которая рассчитывается и должна быть минимальной
                print('MSE test:', round(mean_squared_error(y_test, fcast['mean']), 2))                #ошибка которая рассчитывается и должна быть минимальной


                #FORECASTING
                X_train_f, y_train_f = all_new[independent_cols], all_new['prc_im']
                model_arima = sm.tsa.arima.ARIMA(y_train_f, X_train_f, order=best_o).fit()
                fcast = model_arima.get_forecast(len(X_train_f), exog=X_train_f,
                                                                         alpha=0.05).summary_frame()
                new_ind_1=(fcast['mean'].iloc[:18])
                month=['Jul-2022','Aug-2022','Sep-2022','Okt-2022','Nov-2022','Dec-2022','Jan-2023','Feb-2023','Mar-2023','Apr-2023','May-2023','Jun-2023','Jul-2023','Aug-2023','Sep-2023','Okt-2023','Nov-2023','Dec-2023']

                #create_new_dataset_with_forecasting_data
                forc={"time_2":month,'cnt_ex': fcast['mean'].iloc[:18]}
                forecast=pd.DataFrame(forc)
                # print(forecast)

                #graph plot
                train_forecast = pd.concat([all_new[['time_2','cnt_ex']],forecast],ignore_index=True)
                train_data=train_forecast.iloc[:len(X_train_f)+1]
                forecast_data=train_forecast.iloc[len(X_train_f):len(train_forecast)]
                time_train=train_data["time_2"]
                forecast_train=forecast_data["time_2"]
                plt.figure(figsize=(14,8))
                plt.title(f'Тренувальні дані і прогнозування {name_product}')
                plt.plot('time_2', 'cnt_ex', data=train_data,
                         label='тренувальні дані', linestyle='-', marker='o')
                plt.plot('time_2', 'cnt_ex', data=forecast_data,
                         label='прогноз', linestyle='-', marker='o')
                plt.legend(loc='upper left', fontsize=8)
                plt.xticks(rotation=75)
                plt.show()
                print('Введіть - 0, щоб повторити прогнозування')
                print('Введіть - 1, щоб приступити до кластеризації')
                print("Введіть - 2, щоб дослідити кореляційний зв'язок між показниками")
                print('Введіть - 3, щоб зупинити програму')
                g = int(input('Введіть число='))


            elif ans==2:
            #ARIMA MODEL індекси фізичного обсягу імпорту Ласпейреса
                independent_cols = ['cnt_ex', 'prc_ex', 'prc_im', 'g_qnt', 'g_prc']               # прогнозування фізичного обсягу

                X_train, X_test, y_train, y_test = train_test_split(
                    all_new[independent_cols], all_new['cnt_im'],
                    test_size=0.2, shuffle=False)
                min_1 = 99999999
                min_2=999999999
                #parameters p,q,r
                for p in range(5):
                    for q in range(5):
                            for r in range(5):
                                perm = combinations((p, q, r),3)
                                for i in list(perm):

                                    model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=i).fit()
                                    fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                                         alpha=0.05).summary_frame()
                                    mape = round(mean_absolute_percentage_error(y_test, fcast['mean']), 2)
                                    mse = round(mean_squared_error(y_test, fcast['mean']), 2)
                                    print(i,mape,mse)
                                    if mape<min_1 and mse<min_2:
                                        min_1,min_2,best_o=mape,mse,i
                                        continue
                                    else:
                                        continue
                print('Best ARIMA%s MAPE=%.3f MSE=%.3f' % (best_o, min_1,min_2))


                model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=best_o).fit()                     # вместо 1,3,1 - p,q,r - MAPE и MSE должны быть минимальны
                print(model_arima.summary())
                fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                  alpha=0.05).summary_frame()  # 95% conf
                print('MAPE test:', round(mean_absolute_percentage_error(y_test, fcast['mean']), 2))   #ошибка которая рассчитывается и должна быть минимальной
                print('MSE test:', round(mean_squared_error(y_test, fcast['mean']), 2))                #ошибка которая рассчитывается и должна быть минимальной


                #FORECASTING
                X_train_f, y_train_f = all_new[independent_cols], all_new['cnt_im']
                model_arima = sm.tsa.arima.ARIMA(y_train_f, X_train_f, order=best_o).fit()
                fcast = model_arima.get_forecast(len(X_train_f), exog=X_train_f,
                                                                         alpha=0.05).summary_frame()
                new_ind_1=(fcast['mean'].iloc[:18])
                month=['Jul-2022','Aug-2022','Sep-2022','Okt-2022','Nov-2022','Dec-2022','Jan-2023','Feb-2023','Mar-2023','Apr-2023','May-2023','Jun-2023','Jul-2023','Aug-2023','Sep-2023','Okt-2023','Nov-2023','Dec-2023']

                #create_new_dataset_with_forecasting_data
                forc={"time_2":month,'cnt_im': fcast['mean'].iloc[:18]}
                forecast=pd.DataFrame(forc)
                # print(forecast)

                #graph plot
                train_forecast = pd.concat([all_new[['time_2','cnt_im']],forecast],ignore_index=True)
                train_data=train_forecast.iloc[:len(X_train_f)+1]
                forecast_data=train_forecast.iloc[len(X_train_f):len(train_forecast)]
                time_train=train_data["time_2"]
                forecast_train=forecast_data["time_2"]
                plt.figure(figsize=(14,8))
                plt.title(f'Тренувальні дані і прогнозування {name_product}')
                plt.plot('time_2', 'cnt_im', data=train_data,
                         label='тренувальні дані', linestyle='-', marker='o')
                plt.plot('time_2', 'cnt_im', data=forecast_data,
                         label='прогноз', linestyle='-', marker='o')
                plt.legend(loc='upper left', fontsize=8)
                plt.xticks(rotation=75)
                plt.show()
                print('Введіть - 0, щоб повторити прогнозування')
                print('Введіть - 1, щоб приступити до кластеризації')
                print("Введіть - 2, щоб дослідити кореляційний зв'язок між показниками")
                print('Введіть - 3, щоб зупинити програму')
                g = int(input('Введіть число='))
            elif ans==3:
            #ARIMA MODEL індекc цін експорт Пааше
                independent_cols = ['cnt_ex', 'cnt_im','prc_im', 'g_qnt', 'g_prc']               # прогнозування фізичного обсягу

                X_train, X_test, y_train, y_test = train_test_split(
                    all_new[independent_cols], all_new['prc_ex'],
                    test_size=0.2, shuffle=False)
                min_1 = 99999999
                min_2=999999999
                #parameters p,q,r
                for p in range(5):
                    for q in range(5):
                            for r in range(5):
                                perm = combinations((p, q, r),3)
                                for i in list(perm):

                                    model_arima = sm.tsa.arima.ARIMA(y_train, X_train, i).fit()
                                    fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                                         alpha=0.05).summary_frame()
                                    mape = round(mean_absolute_percentage_error(y_test, fcast['mean']), 2)
                                    mse = round(mean_squared_error(y_test, fcast['mean']), 2)
                                    print(i,mape,mse)
                                    if mape<min_1 and mse<min_2:
                                        min_1,min_2,best_o=mape,mse,i
                                        continue
                                    else:
                                        continue
                print('Best ARIMA%s MAPE=%.3f MSE=%.3f' % (best_o, min_1,min_2))



                model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=best_o).fit()                     # вместо 1,3,1 - p,q,r - MAPE и MSE должны быть минимальны
                print(model_arima.summary())
                fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                  alpha=0.05).summary_frame()  # 95% conf
                print('MAPE test:', round(mean_absolute_percentage_error(y_test, fcast['mean']), 2))   #ошибка которая рассчитывается и должна быть минимальной
                print('MSE test:', round(mean_squared_error(y_test, fcast['mean']), 2))                #ошибка которая рассчитывается и должна быть минимальной


                #FORECASTING
                X_train_f, y_train_f = all_new[independent_cols], all_new['prc_ex']
                model_arima = sm.tsa.arima.ARIMA(y_train_f, X_train_f, best_o).fit()
                fcast = model_arima.get_forecast(len(X_train_f), exog=X_train_f,
                                                                         alpha=0.05).summary_frame()
                new_ind_1=(fcast['mean'].iloc[:18])
                month=['Jul-2022','Aug-2022','Sep-2022','Okt-2022','Nov-2022','Dec-2022','Jan-2023','Feb-2023','Mar-2023','Apr-2023','May-2023','Jun-2023','Jul-2023','Aug-2023','Sep-2023','Okt-2023','Nov-2023','Dec-2023']

                #create_new_dataset_with_forecasting_data
                forc={"time_2":month,'prc_ex': fcast['mean'].iloc[:18]}
                forecast=pd.DataFrame(forc)
                # print(forecast)

                #graph plot
                train_forecast = pd.concat([all_new[['time_2','prc_ex']],forecast],ignore_index=True)
                train_data=train_forecast.iloc[:len(X_train_f)+1]
                forecast_data=train_forecast.iloc[len(X_train_f):len(train_forecast)]
                time_train=train_data["time_2"]
                forecast_train=forecast_data["time_2"]
                plt.figure(figsize=(14,8))
                plt.title(f'Тренувальні дані і прогнозування {name_product}')
                plt.plot('time_2', 'prc_ex', data=train_data,
                         label='тренувальні дані', linestyle='-', marker='o')
                plt.plot('time_2', 'prc_ex', data=forecast_data,
                         label='прогноз', linestyle='-', marker='o')
                plt.legend(loc='upper left', fontsize=8)
                plt.xticks(rotation=75)
                plt.show()
                print('Введіть - 0, щоб повторити прогнозування')
                print('Введіть - 1, щоб приступити до кластеризації')
                print("Введіть - 2, щоб дослідити кореляційний зв'язок між показниками")
                print('Введіть - 3, щоб зупинити програму')
                g = int(input('Введіть число='))
            elif ans==4:
            #ARIMA MODEL індекc цін імпорт Пааше
                independent_cols =  ['cnt_ex', 'cnt_im', 'prc_ex', 'g_qnt', 'g_prc']              # прогнозування фізичного обсягу

                X_train, X_test, y_train, y_test = train_test_split(
                    all_new[independent_cols], all_new['prc_im'],
                    test_size=0.2, shuffle=False)
                min_1 = 99999999
                min_2=999999999
                #parameters p,q,r
                for p in range(5):
                    for q in range(5):
                            for r in range(5):
                                perm = combinations((p, q, r),3)
                                for i in list(perm):

                                    model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=i).fit()
                                    fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                                         alpha=0.05).summary_frame()
                                    mape = round(mean_absolute_percentage_error(y_test, fcast['mean']), 2)
                                    mse = round(mean_squared_error(y_test, fcast['mean']), 2)
                                    print(i,mape,mse)
                                    if mape<min_1 and mse<min_2:
                                        min_1,min_2,best_o=mape,mse,i
                                        continue
                                    else:
                                        continue
                print('Best ARIMA%s MAPE=%.3f MSE=%.3f' % (best_o, min_1,min_2))



                model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=best_o).fit()                     # вместо 1,3,1 - p,q,r - MAPE и MSE должны быть минимальны
                print(model_arima.summary())
                fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                  alpha=0.05).summary_frame()  # 95% conf
                print('MAPE test:', round(mean_absolute_percentage_error(y_test, fcast['mean']), 2))   #ошибка которая рассчитывается и должна быть минимальной
                print('MSE test:', round(mean_squared_error(y_test, fcast['mean']), 2))                #ошибка которая рассчитывается и должна быть минимальной


                #FORECASTING
                X_train_f, y_train_f = all_new[independent_cols], all_new['prc_im']
                model_arima = sm.tsa.arima.ARIMA(y_train_f, X_train_f, order=best_o).fit()
                fcast = model_arima.get_forecast(len(X_train_f), exog=X_train_f,
                                                                         alpha=0.05).summary_frame()
                new_ind_1=(fcast['mean'].iloc[:18])
                month=['Jul-2022','Aug-2022','Sep-2022','Okt-2022','Nov-2022','Dec-2022','Jan-2023','Feb-2023','Mar-2023','Apr-2023','May-2023','Jun-2023','Jul-2023','Aug-2023','Sep-2023','Okt-2023','Nov-2023','Dec-2023']

                #create_new_dataset_with_forecasting_data
                forc={"time_2":month,'prc_im': fcast['mean'].iloc[:18]}
                forecast=pd.DataFrame(forc)
                # print(forecast)

                #graph plot
                train_forecast = pd.concat([all_new[['time_2','prc_im']],forecast],ignore_index=True)
                train_data=train_forecast.iloc[:len(X_train_f)+1]
                forecast_data=train_forecast.iloc[len(X_train_f):len(train_forecast)]
                time_train=train_data["time_2"]
                forecast_train=forecast_data["time_2"]
                plt.figure(figsize=(14,8))
                plt.title(f'Тренувальні дані і прогнозування {name_product}')
                plt.plot('time_2', 'prc_im', data=train_data,
                         label='тренувальні дані', linestyle='-', marker='o')
                plt.plot('time_2', 'prc_im', data=forecast_data,
                         label='прогноз', linestyle='-', marker='o')
                plt.legend(loc='upper left', fontsize=8)
                plt.xticks(rotation=75)
                plt.show()
                print('Введіть - 0, щоб повторити прогнозування')
                print('Введіть - 1, щоб приступити до кластеризації')
                print("Введіть - 2, щоб дослідити кореляційний зв'язок між показниками")
                print('Введіть - 3, щоб зупинити програму')
                g = int(input('Введіть число='))
            elif ans==5:
            #ARIMA MODEL індекc умов торгівлі кількісний
                independent_cols = ['cnt_im', 'prc_ex','prc_im', 'g_prc']              # прогнозування фізичного обсягу

                X_train, X_test, y_train, y_test = train_test_split(
                    all_new[independent_cols], all_new['g_qnt'],
                    test_size=0.2, shuffle=False)
                min_1 = 99999999
                min_2=999999999
                #parameters p,q,r
                for p in range(5):
                    for q in range(5):
                            for r in range(5):
                                perm = combinations((p, q, r),3)
                                for i in list(perm):

                                    model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=i).fit()
                                    fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                                         alpha=0.05).summary_frame()
                                    mape = round(mean_absolute_percentage_error(y_test, fcast['mean']), 2)
                                    mse = round(mean_squared_error(y_test, fcast['mean']), 2)
                                    print(i,mape,mse)
                                    if mape<min_1 and mse<min_2:
                                        min_1,min_2,best_o=mape,mse,i
                                        continue
                                    else:
                                        continue
                print('Best ARIMA%s MAPE=%.3f MSE=%.3f' % (best_o, min_1,min_2))



                model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=best_o).fit()                     # вместо 1,3,1 - p,q,r - MAPE и MSE должны быть минимальны
                print(model_arima.summary())
                fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                  alpha=0.05).summary_frame()  # 95% conf
                print('MAPE test:', round(mean_absolute_percentage_error(y_test, fcast['mean']), 2))   #ошибка которая рассчитывается и должна быть минимальной
                print('MSE test:', round(mean_squared_error(y_test, fcast['mean']), 2))                #ошибка которая рассчитывается и должна быть минимальной


                #FORECASTING
                X_train_f, y_train_f = all_new[independent_cols], all_new['g_qnt']
                model_arima = sm.tsa.arima.ARIMA(y_train_f, X_train_f, order=best_o).fit()
                fcast = model_arima.get_forecast(len(X_train_f), exog=X_train_f,
                                                                         alpha=0.05).summary_frame()
                new_ind_1=(fcast['mean'].iloc[:18])
                month=['Jul-2022','Aug-2022','Sep-2022','Okt-2022','Nov-2022','Dec-2022','Jan-2023','Feb-2023','Mar-2023','Apr-2023','May-2023','Jun-2023','Jul-2023','Aug-2023','Sep-2023','Okt-2023','Nov-2023','Dec-2023']

                #create_new_dataset_with_forecasting_data
                forc={"time_2":month,'g_qnt': fcast['mean'].iloc[:18]}
                forecast=pd.DataFrame(forc)
                # print(forecast)

                #graph plot
                train_forecast = pd.concat([all_new[['time_2','g_qnt']],forecast],ignore_index=True)
                train_data=train_forecast.iloc[:len(X_train_f)+1]
                forecast_data=train_forecast.iloc[len(X_train_f):len(train_forecast)]
                time_train=train_data["time_2"]
                forecast_train=forecast_data["time_2"]
                plt.figure(figsize=(14,8))
                plt.title(f'Тренувальні дані і прогнозування {name_product}')
                plt.plot('time_2', 'g_qnt', data=train_data,
                         label='тренувальні дані', linestyle='-', marker='o')
                plt.plot('time_2', 'g_qnt', data=forecast_data,
                         label='прогноз', linestyle='-', marker='o')
                plt.legend(loc='upper left', fontsize=8)
                plt.xticks(rotation=75)
                plt.show()
                print('Введіть - 0, щоб повторити прогнозування')
                print('Введіть - 1, щоб приступити до кластеризації')
                print("Введіть - 2, щоб дослідити кореляційний зв'язок між показниками")
                print('Введіть - 3, щоб зупинити програму')
                g = int(input('Введіть число='))
            elif ans == 6:
                # ARIMA MODEL індекc умов торгівлі кількісний
                independent_cols = ['cnt_ex', 'cnt_im', 'prc_ex','prc_im', 'g_qnt']  # прогнозування фізичного обсягу

                X_train, X_test, y_train, y_test = train_test_split(
                    all_new[independent_cols], all_new['g_prc'],
                    test_size=0.2, shuffle=False)
                min_1 = 99999999
                min_2 = 999999999
                # parameters p,q,r
                for p in range(5):
                    for q in range(5):
                        for r in range(5):
                            perm = combinations((p, q, r), 3)
                            for i in list(perm):

                                model_arima = sm.tsa.arima.ARIMA(y_train, X_train, order=i).fit()
                                fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                                 alpha=0.05).summary_frame()
                                mape = round(mean_absolute_percentage_error(y_test, fcast['mean']), 2)
                                mse = round(mean_squared_error(y_test, fcast['mean']), 2)
                                print(i, mape, mse)
                                if mape < min_1 and mse < min_2:
                                    min_1, min_2, best_o = mape, mse, i
                                    continue
                                else:
                                    continue
                print('Best ARIMA%s MAPE=%.3f MSE=%.3f' % (best_o, min_1, min_2))

                model_arima = sm.tsa.arima.ARIMA(y_train, X_train,
                                                 order=best_o).fit()  # вместо 1,3,1 - p,q,r - MAPE и MSE должны быть минимальны
                print(model_arima.summary())
                fcast = model_arima.get_forecast(len(X_test), exog=X_test,
                                                 alpha=0.05).summary_frame()  # 95% conf
                print('MAPE test:', round(mean_absolute_percentage_error(y_test, fcast['mean']),
                                          2))  # ошибка которая рассчитывается и должна быть минимальной
                print('MSE test:', round(mean_squared_error(y_test, fcast['mean']),
                                         2))  # ошибка которая рассчитывается и должна быть минимальной

                # FORECASTING
                X_train_f, y_train_f = all_new[independent_cols], all_new['g_prc']
                model_arima = sm.tsa.arima.ARIMA(y_train_f, X_train_f, order=best_o).fit()
                fcast = model_arima.get_forecast(len(X_train_f), exog=X_train_f,
                                                 alpha=0.05).summary_frame()
                new_ind_1 = (fcast['mean'].iloc[:18])
                month = ['Jul-2022', 'Aug-2022', 'Sep-2022', 'Okt-2022', 'Nov-2022', 'Dec-2022', 'Jan-2023', 'Feb-2023',
                         'Mar-2023', 'Apr-2023', 'May-2023', 'Jun-2023', 'Jul-2023', 'Aug-2023', 'Sep-2023', 'Okt-2023',
                         'Nov-2023', 'Dec-2023']

                # create_new_dataset_with_forecasting_data
                forc = {"time_2": month, 'g_prc': fcast['mean'].iloc[:18]}
                forecast = pd.DataFrame(forc)
                # print(forecast)

                # graph plot
                train_forecast = pd.concat([all_new[['time_2', 'g_prc']], forecast], ignore_index=True)
                train_data = train_forecast.iloc[:len(X_train_f) + 1]
                forecast_data = train_forecast.iloc[len(X_train_f):len(train_forecast)]
                time_train = train_data["time_2"]
                forecast_train = forecast_data["time_2"]
                plt.figure(figsize=(14, 8))
                plt.title(f'Тренувальні дані і прогнозування {name_product}')
                plt.plot('time_2', 'g_prc', data=train_data,
                         label='тренувальні дані', linestyle='-', marker='o')
                plt.plot('time_2', 'g_prc', data=forecast_data,
                         label='прогноз', linestyle='-', marker='o')
                plt.legend(loc='upper left', fontsize=8)
                plt.xticks(rotation=75)
                plt.show()
                print('Введіть - 0, щоб повторити прогнозування')
                print('Введіть - 1, щоб приступити до кластеризації')
                print("Введіть - 2, щоб дослідити кореляційний зв'язок між показниками")
                print('Введіть - 3, щоб зупинити програму')
                g = int(input('Введіть число='))

        else:
                print('На жаль неможливо зробити прогноз по цій групі товарів, через те що забагато даних пропущено')
                print('Введіть - 0, щоб повторити прогнозування')
                print('Введіть - 1, щоб приступити до кластеризації')
                print("Введіть - 2, щоб дослідити кореляційний зв'язок між показниками")
                print('Введіть - 3, щоб зупинити програму')
                g = int(input('Введіть число='))
# кластеризація
while g == 1:
                    df1 = df[['time_2', 'g_qnt', 'g_prc', 'group']]

                    empty_dataset = pd.DataFrame(columns=['time_2', 'g_qnt', 'g_prc', 'group'])
                    product = ['01 живi тварини', '02 м’ясо та їстівні субпродукти', '03 риба i ракоподібні',
                               '04 молоко та молочнi продукти яйця птиці; натуральний мед', '05 інші продукти тваринного походження',
                               '06 живі дерева та інші рослини', '07 овочі', '08 їстівні плоди та горіхи', '09 кава, чай',
                               '10 зерновi культури', '11 продукцiя борошномельно-круп’яної промисловості',
                               '12 насiння і плоди олійних рослин',
                               '13 шелак природний', '14 рослинні матеріали для виготовлення',
                               '15 жири та олії тваринного або рослинного походження', '16 продукти з м’яса, риби',
                               '17 цукор і кондитерські вироби з цукру',
                               '18 какао та продукти з нього', '19 готові продукти із зерна', '20 продукти переробки овочів',
                               '21 різні харчовi продукти', '22 алкогольнi i безалкогольнi напої та оцет',
                               '23 залишки і вiдходи харчової промисловості', '24 тютюн і промислові замінники тютюну',
                               '25 сiль; сірка; землі та каміння', '26 руди, шлак і зола',
                               '27 палива мінеральні; нафта і продукти її перегонки',
                               '28 продукти неорганiчної хімії', '29 органiчнi хiмiчнi сполуки', '30 фармацевтична продукція',
                               '31 добрива', '32 екстракти дубильні', '33 ефiрнi олії',
                               '34 мило, поверхнево-активні органічні речовини',
                               '35 бiлковi речовини', '36 порох і вибуховi речовини', '37 фотографічні або кiнематографічні товари',
                               '38 різноманітна хімічна продукція', '39 пластмаси, полімерні матеріали', '40 каучук, гума',
                               '41 шкури', '42 вироби із шкіри', '43 натуральне та штучне хутро', '44 деревина і вироби з деревини',
                               '45 корок та вироби з нього', '46 вироби із соломи', '47 маса з деревини', '48 папiр та картон',
                               '49 друкована продукція', '50 шовк', '51 вовна', '52 бавовна', '53 іншi текстильнi волокна',
                               '54 нитки синтетичні або штучні', '55 синтетичні або штучні штапельнi волокна',
                               '56 вата', '57 килими', '58 спецiальнi тканини', '59 текстильнi матеріали', '60 трикотажні полотна',
                               '61 одяг та додаткові речі до одягу, трикотажні', '62 одяг та додаткові речі до одягу, текстильні',
                               '63 іншi готовi текстильні вироби', '64 взуття', '65 головнi убори', '66 парасольки',
                               '67 обробленi пір’я та пух', '68 вироби з каменю, гiпсу, цементу', '69 керамiчнi вироби',
                               '70 скло та вироби із скла',
                               '71 перли природні або культивовані, дорогоцінне або напівдорогоцінне каміння', '72 чорнi метали',
                               '73 вироби з чорних металів', '74 мiдь i вироби з неї', '75 нiкель i вироби з нього',
                               '76 алюмiнiй i вироби з нього', '77 свинець і вироби з нього', '78 цинк i вироби з нього',
                               '79 олово і вироби з нього', '80 іншi недорогоцінні метали', '81 інструменти, ножовi вироби',
                               '82 іншi вироби з недорогоцінних металiв', '83 реактори ядерні, котли, машини', '84 електричнi машини',
                               '85 залізничні локомотиви', '86 засоби наземного транспорту, крiм залізничного',
                               '87 літальні апарати', '88 судна', '89 прилади та апарати оптичнi, фотографічні', '90 годинники',
                               '91 музичні інструменти', '92 меблi', '93 іграшки', '94 рiзнi готовi вироби', '95 твори мистецтва'
                               ]

                    i = 0

                    for i in range(96):  # цикл формування датасету із груп товарів даних про які допустима кількість
                        clus_data = df1[df1['group'] == i + 1]
                        q1 = clus_data['g_qnt'].isnull().sum()
                        q2 = clus_data['g_prc'].isnull().sum()
                        if q1 < 5 and q2 < 5:
                            empty_dataset = pd.concat([empty_dataset, clus_data], ignore_index=True)
                            i = +1
                    print("Кластеризація груп товарів")
                    print("Місяці надзвичайних подій")
                    print("COVID-19 - Apr-2019")
                    print("Аварія на Босфорі - May-2019")
                    print("Формування 20- го Кабінету Міністрів - Aug-2019")
                    print("Формування 21- го Кабінету Міністрів - Mar-2020")
                    print("Війна в Україні - Mar-2022")
                    b = str(input(" Місяць надзвичайної події по якому проводиться кластеризація = "))

                    new_interp = empty_dataset.interpolate(method='polynomial', order=2)
                    w = 0

                    print(new_interp)
                    this_month = new_interp[new_interp['time_2'] == b]

                    if b == "Apr-2019":
                        previous_month = "Mar-2019"
                        before_month = new_interp[new_interp['time_2'] == previous_month]
                    elif b == "May-2019":
                        previous_month = "Apr-2019"
                        before_month = new_interp[new_interp['time_2'] == previous_month]
                    elif b == "Aug-2019":
                        previous_month = "Jul-2019"
                        before_month = new_interp[new_interp['time_2'] == previous_month]
                    elif b == "Mar-2020":
                        previous_month = "Feb-2020"
                        before_month = new_interp[new_interp['time_2'] == previous_month]
                    elif b == "Mar-2022":
                        previous_month = "Feb-2022"
                        before_month = new_interp[new_interp['time_2'] == previous_month]

                    this_month_s = (this_month.reset_index())[['g_qnt', 'g_prc']]
                    before_month_s = (before_month.reset_index())[['g_qnt', 'g_prc']]
                    coef_0 = this_month_s / before_month_s  # датасет без дат і груп товарів

                    coef_1 = {'g_qnt': coef_0.g_qnt, 'g_prc': coef_0.g_prc}

                    coef = pd.DataFrame(coef_1).iloc[:, [0, 1]].values

                    # # print(before_month[['g_qnt','g_prc']]-this_month[['g_qnt','g_prc']])

                    import scipy.cluster.hierarchy as sch

                    dendrogram = sch.dendrogram(sch.linkage(coef, method='ward'))
                    plt.title('Дендограма')
                    plt.xlabel('Індекс товарів')
                    plt.ylabel('Евклідові відстані')
                    plt.show()

                    plt.show()

                    from sklearn.cluster import AgglomerativeClustering

                    X = coef
                    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
                    y_hc = cluster.fit_predict(coef)

                    gr = this_month['group'].reset_index()

                    res_group = (gr['group'])

                    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Кластер груп товарів №1 ')
                    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Кластер груп товарів №2')
                    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Кластер груп товарів №3')
                    for f in range(0, res_group.shape[0]):
                        plt.annotate(str(res_group[f]), xy=(coef[f, 0], coef[f, 1]), xytext=(1, 1), textcoords='offset points')
                    plt.title(f'Кластери груп товарів за впливом надзвичайної умови у місяць {b}')
                    plt.xlabel('Індекс кількісних умов торгівлі')
                    plt.ylabel('Індекс цінових умов торгівлі')
                    plt.legend()
                    plt.show()
                    print("Введіть - 1, щоб повторити кластеризацію")
                    print("Введіть - 2, щоб приступити до дослідження кореляційного зв'язку між показниками")
                    print('Введіть - 3, щоб зупинити програму')
                    g = int(input('Введіть число='))

while g==2:
            print("Дослідження кореляції між показниками ефективності в окремий період")
            print("Місяці надзвичайних подій")
            print("COVID-19 - Apr-2019")
            print("Аварія на Босфорі - May-2019")
            print("Формування 20- го Кабінету Міністрів - Aug-2019")
            print("Формування 21- го Кабінету Міністрів - Mar-2020")
            print("Війна в Україні - Mar-2022")
            mon = str(input(" Введіть місяць та рік надзвичайної події, або будь-який інший місяць та рік із проміжку 2017-2022 у який досліджується кореляція = "))

            cols = ['cnt_ex', 'cnt_im', 'prc_ex', 'prc_im', 'g_qnt', 'g_prc']
            all_new = df.interpolate(method='polynomial', order=2)
            all_corr=all_new[all_new['time_2']==mon]
            corr = all_corr[cols].corr().round(2)

            plt.figure(figsize=(7, 7))
            plt.title(f'Кореляційна матриця між показниками ефективності за місяць {mon}')
            sns.heatmap(corr, square=True, cmap='Spectral', annot=True)
            plt.xticks(rotation=70)
            plt.show()
            print("Введіть - 2, щоб повторити дослідження кореляційного зв'язку між показниками")
            print('Введіть - 3, щоб зупинити програму')
            g = int(input('Введіть число='))

while g!=0 and g!=1 and g!=2 and g!=3:
            print("Збій програми, бо ви ввели дивний варіант, якого вам не пропонували")
            break



while g==3:
    break





