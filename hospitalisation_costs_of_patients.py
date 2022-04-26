import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

uploaded_file = st.file_uploader("Wprowadź odpowiedni plik")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df = df[['ID_PACJENT', 'TRYB_PRZYJECIA_KOD', 'TYP_KOMORKI_ORG', 'ROZP_GLOWNE', 'MC_SPRAWOZDAWCZY', 'ROK_SPRAWOZDAWCZY','WARTOSC_SKOR']].groupby(
        ['ID_PACJENT', 'TRYB_PRZYJECIA_KOD', 'TYP_KOMORKI_ORG', 'ROZP_GLOWNE', 'MC_SPRAWOZDAWCZY','ROK_SPRAWOZDAWCZY']).sum().reset_index()
    df = df.drop('ID_PACJENT', axis=1)

    oddzialy = df.TYP_KOMORKI_ORG.sort_values().unique().tolist()
    oddzialy.insert(0, 'Wybierz oddzial')
    koszyki = ['Wybierz wartosc', 2, 3, 4, 5]

    with st.sidebar.header('1. Wybierz numer oddziału dla którego chcesz wykonać predykcję:'):
        TYP_KOMORKI_ORG = st.sidebar.selectbox('Wybierz numer oddziału:', oddzialy, index=0)

    if TYP_KOMORKI_ORG != 'Wybierz oddzial':

        df1 = df[df['TYP_KOMORKI_ORG'] == TYP_KOMORKI_ORG].reset_index(drop=True)
        st.info("Wielkość tabeli danych dla wybranego oddziału: **{}**" .format(df1.shape))
        st.markdown('**Fragment badanej tabeli danych: **')
        st.dataframe(df1.head())

        with st.sidebar.header('2. Wybierz liczbę przedziałów wartości kosztów, jaką chcesz utworzyć:'):
            LICZBA_KOSZYKOW = st.sidebar.selectbox('Wybierz liczbę:', koszyki, index=0)

        if LICZBA_KOSZYKOW != 'Wybierz wartosc':

            przedzialy = pd.qcut(df1['WARTOSC_SKOR'], q=LICZBA_KOSZYKOW)
            df1['WARTOSC_SKOR'] = pd.qcut(df1['WARTOSC_SKOR'], q=LICZBA_KOSZYKOW, labels=np.arange(LICZBA_KOSZYKOW))

            etykiety_przedzialow = df1.WARTOSC_SKOR.value_counts().reset_index().rename(columns={'index':'Etykiety_przedziałów'})
            przedzialy = pd.DataFrame(przedzialy.value_counts().reset_index().rename(columns={'index':'Przedziały_kosztów'}))
            przedzialy['Przedziały_kosztów'] = przedzialy['Przedziały_kosztów'].astype('string')
            przedzialy = pd.merge(przedzialy,etykiety_przedzialow)

            st.markdown('**Tabela ukazująca utworzone przedziały kosztów dla badanego oddziału: **')
            st.dataframe(przedzialy.rename(columns={'WARTOSC_SKOR':'Liczba_wartości_w_przedziale'}).sort_values('Etykiety_przedziałów'))
            st.markdown('**Fragment badanej tabeli danych z zakodowaną etykietą dla ''WARTOSC_SKOR'' na podstawie zdefiniowanych przedziałów: **')
            st.dataframe(df1.head())

            if ((LICZBA_KOSZYKOW != 'Wybierz wartosc')&(TYP_KOMORKI_ORG != 'Wybierz oddzial')):

                with st.sidebar.header('3. Wybierz algortym uczenia maszynowego, za pomocą którego chesz dokonać predykcji:'):
                    MODELE = st.sidebar.selectbox('Wybierz algortym:', ['Wybierz algorytm', 'DecisionTree', 'RandomForest', 'KNN', 'NaiveBayes'], index=0)

                if MODELE != 'Wybierz algorytm':
                    if MODELE == 'DecisionTree':
                        clf = DecisionTreeClassifier()
                    elif MODELE == 'RandomForest':
                        clf = RandomForestClassifier()
                    elif MODELE == 'NaiveBayes':
                        clf = BernoulliNB()
                    else:
                        clf = KNeighborsClassifier()

                    Y = df1.pop('WARTOSC_SKOR')
                    X = pd.get_dummies(df1)
                    st.info(f'Klasyfikator = **{MODELE}**')

                    with st.sidebar.header('4. Czy chcesz podzielić dane na treningowe i testowe, oraz nauczyć i przetestować na nich wybrany algorytm?'):
                        check1 = st.sidebar.checkbox('TAK')
                        check2 = st.sidebar.checkbox('NIE')

                    if check1:

                        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y)
                        st.info('Ilość rekordów w zbiorze treningowym: **{}**  Ilość rekordów w zbiorze testowym: **{}**'.format(len(X_train), len(X_test)))
                        y_tr_l = y_train.value_counts().reset_index().rename(columns={'index':'Etykiety','WARTOSC_SKOR':'Ilośc etykiet w danych treningowych'})
                        t_te_l =y_test.value_counts().reset_index().rename(columns={'index':'Etykiety','WARTOSC_SKOR':'Ilośc etykiet w danych testowych'})
                        st.dataframe(pd.merge(y_tr_l,t_te_l))
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = np.round(accuracy_score(y_test, y_pred), 3)
                        f1score = np.round(f1_score(y_test, y_pred, average='weighted'), 3)

                        st.markdown('**Metryki jakości klasyfikacji uzyskane przez model na danych testowych**:')
                        col1, col2, col3 = st.columns(3)
                        col1.metric('Model', MODELE)
                        col2.metric('Accuracy: ', acc)
                        col3.metric('F1-score: ',f1score)

                    if check2:
                            st.info('Zakończyłeś korzystanie z aplikacji')

                    if ((TYP_KOMORKI_ORG != 'Wybierz oddzial')&(LICZBA_KOSZYKOW != 'Wybierz wartosc')&(MODELE != 'Wybierz algorytm')&check1):

                        ROZP_GLOWNE_labels = df1.ROZP_GLOWNE.sort_values().unique().tolist()
                        ROZP_GLOWNE_labels.insert(0, 'Wybierz wartosc')
                        TRYB_PRZYJECIA_KOD_labels = df1.TRYB_PRZYJECIA_KOD.sort_values().unique().tolist()
                        TRYB_PRZYJECIA_KOD_labels.insert(0, 'Wybierz wartosc')
                        MC_SPRAWOZDAWCZY_labels = df1.MC_SPRAWOZDAWCZY.sort_values().unique().tolist()
                        MC_SPRAWOZDAWCZY_labels.insert(0, 'Wybierz wartosc')
                        ROK_SPRAWOZDAWCZY_labels = df1.ROK_SPRAWOZDAWCZY.sort_values().unique().tolist()
                        ROK_SPRAWOZDAWCZY_labels.insert(0, 'Wybierz wartosc')

                        st.subheader('5. Wybierz parametry pacjenta, dla którego ma zostać wykonana predykcja sumy kosztów, przy użyciu wytrenowanego modelu: ')
                        form = st.form(key='my_form')
                        TRYB_PRZYJECIA_KOD = form.selectbox('TRYB_PRZYJECIA_KOD', TRYB_PRZYJECIA_KOD_labels, index=0)
                        ROZP_GLOWNE = form.selectbox('ROZP_GLOWNE', ROZP_GLOWNE_labels, index=0)
                        MC_SPRAWOZDAWCZY = form.selectbox('MC_SPRAWOZDAWCZY', MC_SPRAWOZDAWCZY_labels, index=0)
                        ROK_SPRAWOZDAWCZY = form.selectbox('ROK_SPRAWOZDAWCZY', ROK_SPRAWOZDAWCZY_labels, index=0)
                        submit_button = form.form_submit_button(label='Wykonaj')
                        if ((ROZP_GLOWNE != 'Wybierz wartosc')&(TRYB_PRZYJECIA_KOD != 'Wybierz wartosc')&(MC_SPRAWOZDAWCZY != 'Wybierz wartosc')
                            &(ROK_SPRAWOZDAWCZY != 'Wybierz wartosc')&(MODELE != 'Wybierz algorytm')&check1):

                            data ={'TRYB_PRZYJECIA_KOD': TRYB_PRZYJECIA_KOD,
                                    'TYP_KOMORKI_ORG': TYP_KOMORKI_ORG,
                                    'ROZP_GLOWNE': ROZP_GLOWNE,
                                    'MC_SPRAWOZDAWCZY': MC_SPRAWOZDAWCZY,
                                    'ROK_SPRAWOZDAWCZY': ROK_SPRAWOZDAWCZY}

                            features = pd.DataFrame(data, index=[0])
                            dff = pd.concat([features, df1], axis=0)
                            dff = pd.get_dummies(dff)
                            dff = dff[0:1]
                            df_values = pd.DataFrame()
                            for col in dff.columns[1:]:
                                df_values = df_values.append(dff.loc[dff[col] != 0, col])
                            df_values = df_values.dropna().T
                            df_values.insert(0,'TRYB_PRZYJECIA_KOD',TRYB_PRZYJECIA_KOD)
                            st.subheader('Wybrane przez użytkownika parametry na podstawie których przewidziano przedział kosztów hospitalizacji pacjenta')
                            st.table(df_values)
                            prediction = clf.predict(dff)
                            prediction = int(np.array(prediction))
                            wartosc_predykcji = przedzialy.loc[przedzialy['Etykiety_przedziałów']==prediction]['Przedziały_kosztów']
                            st.subheader('Przewidziany przedział sumy kosztów hospitalizacji dla nowego pacjenta na podstawie wybranych parametrów:')
                            st.write(wartosc_predykcji)
                        else:
                            st.warning('Wprowadz wartość wszystkich parametrów')
                else:
                    st.warning('Wybierz model klasyfikacyjny')
            else:
                st.warning("Wybierz model klasyfikacyjny")
        else:
            st.warning('Proszę wybrać przedział z listy')
    else:
        st.warning('Proszę wybrać oddział z listy')
else:
    st.warning('Proszę zamieścić plik, aby skorzystać z aplikacji')