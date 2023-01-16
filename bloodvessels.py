import cv2 
import numpy as np
import os
import csv


def extract_bv(image):	 	
	b,green_fundus,r = cv2.split(image) #yeşil kan damarlarını ayırmak için
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #CLAHE'nin parametreleri
	contrast_enhanced_green_fundus = clahe.apply(green_fundus) #CLAHE uygulanmış yeşil kan damarları

	# applying alternate sequential filtering (3 times closing opening)
	# CV2 Morphological Processing 
	r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1) #morphologyEx fonksiyonu ile açma ve kapatma işlemleri uygulanmıştır. 
	R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1) #getStructuringElement fonksiyonu ile elips şeklinde bir yapı oluşturulmuştur.
	r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1) #iterations parametresi ile açma ve kapatma işlemleri kaç kez tekrar edileceği belirtilmiştir.
	R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1) #5x5 ve 11x11 boyutunda elips yapı oluşturulmuştur.
	r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1) #23x23 boyutunda elips yapı oluşturulmuştur.
	R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1) #morphologyEx fonksiyonu ile açma ve kapatma işlemleri uygulanmıştır.
	f4 = cv2.subtract(R3,contrast_enhanced_green_fundus) #subtract fonksiyonu ile açma ve kapatma işlemleri uygulanmıştır.
	f5 = clahe.apply(f4) #CLAHE uygulanmıştır.

	#Çok küçük konturları alan parametresi ile gürültü kaldırma.
	ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY) #threshold fonksiyonu ile gürültü kaldırma işlemi uygulanmıştır.
	mask = np.ones(f5.shape[:2], dtype="uint8") * 255	 #mask oluşturma
	contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #findContours fonksiyonu ile konturlar bulunmuştur.

	for cnt in contours: #konturların alanı 200'den küçük olanlar silinmiştir.

		if cv2.contourArea(cnt) <= 200: #contourArea fonksiyonu ile konturların alanı hesaplanmıştır.
			cv2.drawContours(mask, [cnt], -1, 0, -1) #drawContours fonksiyonu ile konturlar silinmiştir.
					
	im = cv2.bitwise_and(f5, f5, mask=mask) #bitwise_and fonksiyonu ile mask uygulanmıştır.
	ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV) #threshold fonksiyonu ile gürültü kaldırma işlemi uygulanmıştır.			
	newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	#erode fonksiyonu ile gürültü kaldırma işlemi uygulanmıştır.

	#Kan damarları gibi düz olmayan büyük parçaları silme
	# Kan damarları ve alan aralığı arasında da dahil olmak üzere büyük parçaları silme işlemi uygulanmıştır.
	fundus_eroded = cv2.bitwise_not(newfin)	 #bitwise_not fonksiyonu ile gürültü kaldırma işlemi uygulanmıştır.
	xmask = np.ones(fundus.shape[:2], dtype="uint8") * 255 #xmask oluşturma
	xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #findContours fonksiyonu ile konturlar bulunmuştur.

	for cnt in xcontours: 
		shape = "unidentified" #konturların şekli tanımlanmıştır.
		peri = cv2.arcLength(cnt, True) #arcLength fonksiyonu ile konturların çevresi hesaplanmıştır.
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)    #approxPolyDP fonksiyonu ile konturların yaklaşık şekli hesaplanmıştır.	

		if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100: #konturların alanı 3000'den küçük ve 100'den büyük olanlar silinmiştir.
			shape = "circle"	 #konturların şekli tanımlanmıştır.

		else: #konturların alanı 3000'den küçük ve 100'den büyük olmayanlar silinmemiştir.
			shape = "veins" 

		if(shape=="circle"): 
			cv2.drawContours(xmask, [cnt], -1, 0, -1)	 
	
	finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask) 	
	blood_vessels = cv2.bitwise_not(finimage) #bitwise_not fonksiyonu ile gürültü kaldırma işlemi uygulanmıştır.
	return blood_vessels	 

if __name__ == "__main__":	 #main fonksiyonu tanımlanmıştır.
	pathFolder = r"C:\Users\kaan.gurgen\Desktop\retina-features-master\tester\other_data" 
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))] 
	destinationFolder = "destination"

	if not os.path.exists(destinationFolder):  #destination klasörü oluşturma
		os.mkdir(destinationFolder) #mkdir fonksiyonu ile destination klasörü oluşturulmuştur.

	for file_name in filesArray: #dosyaların isimlerini almak için döngü oluşturulmuştur.
		file_name_no_extension = os.path.splitext(file_name)[0] #dosyaların uzantılarını almak için döngü oluşturulmuştur.
		fundus = cv2.imread(pathFolder+'/'+file_name) #imread fonksiyonu ile dosyaların okunması sağlanmıştır.
		bloodvessel = extract_bv(fundus) #extract_bv fonksiyonu ile kan damarları çıkarılmıştır.
		cv2.imwrite(destinationFolder+file_name_no_extension+"_bloodvessel.png",bloodvessel) #imwrite fonksiyonu ile kan damarları kaydedilmiştir.
    
