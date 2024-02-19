import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

class App:
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.Init_Streamlit_Page()

        self.params = dict()
        self.clf = None
        self.X, self.y = None, None

    def run(self):
        self.get_dataset()

    def Init_Streamlit_Page(self):
        st.title('BAYKAR-CEZERI MODUL PROJESI')
        st.subheader("#Ertuğrul Sert sertugrul18@hotmail.com")
        st.write("""
        Hangi Dataset için hangi classifier daha iyi keşfedelim.
        SVM , KNN , Naive-Bayes""")

    def get_dataset(self):
        data = None
        upload_file = st.sidebar.file_uploader("Veri Seti Seçiniz...", type=['csv'])
        if upload_file is not None:
            data = pd.read_csv(upload_file)
            self.process_data(data)
        else:
            st.error("Bir Veri Seti yükleyiniz...!")

    def process_data(self, data):

        #ilk 10 satırı göster
        st.header("Veri Setinin ilk 10 satırı:")
        st.write( data.head(10))
        st.header("Veri Sütun İsimleri:")
        st.write( data.columns.tolist())

        # Gereksiz sütunları düşürme
        data = data.drop(columns=['Unnamed: 32', 'id'], axis=1, errors='ignore')
        st.header("Veri Setinin son 10 satırı:(id ve Unmaned droplandi)")
        st.write(data.tail(10))
        
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        # Y etiket verisi olarak diagnosis sütununu ayırma
        Y = data['diagnosis']

        # X öznitelik verisi olarak diagnosis dışındaki sütunları kullanma
        X = data.drop('diagnosis', axis=1)

        # Normalize Min max ile SVC uzun suruyor çünkü
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        X = pd.DataFrame(X_normalized, columns=X.columns)

        # Scatter plot oluşturma korelasyon
        st.header("Korelasyon Grafiği")
        fig, ax = plt.subplots()
        sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=data, palette=['green', 'red'], alpha=0.6)
        ax.set_title('Radius Mean vs Texture Mean')
        ax.set_xlabel('Radius Mean')
        ax.set_ylabel('Texture Mean')
        ax.legend(title='Teşhis', labels=['Kötü Huylu', 'İyi Huylu'])
        st.pyplot(fig)

        st.header("Correlation Heatmap(Korelasyon Isı Haritası)")
        corr_matrix = data.corr()  #Korelasyon Hesaplaması
        fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the size as needed
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

        # Veriyi %80 eğitim ve %20 test olacak şekilde ayırma
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



        # Model seçimi için Streamlit sidebar widget'ı
        classifier_name = st.sidebar.selectbox("Classifier", ("KNN", "SVM", "Naïve Bayes"))

        # Model parametrelerini ve GridSearch için parametre grid'ini tanımla
        def get_classifier(clf_name):
            if clf_name == "KNN":
                clf = KNeighborsClassifier()
                parameters = {'n_neighbors': list(range(1, 25)), 'metric': ['euclidean', 'manhattan']}
            elif clf_name == "SVM":
                clf = SVC()
                parameters = {'C': [0.001,0.01,0.1, 1, 5, 10,15,20,25,30,40,50,60,70,80,90,100], 'kernel': ['rbf', 'linear']}
            elif clf_name == "Naïve Bayes":
                clf = GaussianNB()
                parameters = {}  # Naive Bayes için özel bir parametre gridi tanımlamıyoruz
            else:
                st.error("Unknown classifier")
            return clf, parameters

        # Seçilen classifier ve parametre grid'i al
        clf, params = get_classifier(classifier_name)

        # GridSearchCV kullanarak en iyi parametreleri bul
        grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, Y_train)

        # En iyi parametreleri ve skoru göster
        st.header("GridSearchCV ile tahmin")
        st.write(f"Best parameters:(En iyi tahmin Parametere ) {grid_search.best_params_}")
        st.write(f"Best score:(En iyi skor) {grid_search.best_score_:.4f}")

        # Optimum parametrelerle modeli eğit
        best_clf = grid_search.best_estimator_


        # Tahmin Yapma
        Y_pred = best_clf.predict(X_test)

        # Metrics Hesasplamaları
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)
        conf_matrix = confusion_matrix(Y_test, Y_pred)

        # Metricsler
        st.header("Metrics(ölçümler)")
        
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

        # Confisuon matrix
        st.header("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        