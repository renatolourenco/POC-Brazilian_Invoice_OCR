import cv2
import json
import os
import pytesseract
import re
import shutil
import time
from datetime import datetime
from fuzzywuzzy import fuzz
from pdf2image import convert_from_path


def convertPDF2Image(file_name):
    print(fr'Convert PDF to Image Started - File {file_name}')
    start = time.time()
    pages = convert_from_path(fr'./input/{file_name}' + '.pdf', 500)
    for idx, page in enumerate(pages):
        page.save(fr'./processing/{file_name}/{file_name}' + '.png', 'PNG')
        break
    end = time.time()
    print(fr'Convert PDF to Image Finished - File {file_name} === {end - start}')
    markRegion(fr'./processing/{file_name}', file_name)


def extractContours(contours, image, dest_path, image_name):
    print(fr'Extract Contours started - File {file_name}')
    start = time.time()
    roi_list = []
    for idx, c in enumerate(contours):
        perimeter = cv2.arcLength(c, True)
        if 1000 < perimeter < 10000:
            approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
            if len(approx) <= 5:
                (x, y, lar, alt) = cv2.boundingRect(c)
                raw_roi = image[y - 25:(y + alt) + 25, x - 25:(x + lar) + 25]

                (thresh, roi_final_bin) = cv2.threshold(raw_roi, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                roi_list.append(fr'{dest_path}/roi_{idx + 1}.png')
                cv2.imwrite(fr'{dest_path}/roi_{idx + 1}.png', roi_final_bin)
                cv2.rectangle(image, (x, y), (x + lar, y + alt), (0, 255, 0), 6)
    end = time.time()
    print(fr'Extract Contours Finished - File {file_name} === {end - start}')
    extractTxtFromImage(roi_list, image_name)


def markRegion(main_path, image_name):
    print(fr'Mark Regions Started - File {file_name}')
    start = time.time()

    alpha = 0.5
    beta = 1.0 - alpha

    img = cv2.imread(fr'{main_path}/{image_name}.png', 0)

    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=13)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=9, borderValue=10)

    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=13)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=3)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_final_bin = cv2.dilate(
        ~img_final_bin,
        cv2.getStructuringElement(cv2.MORPH_ERODE, (5, 5)),
        iterations=11
    )

    img_final_bin = cv2.GaussianBlur(~img_final_bin, (5, 5), 5)

    img_final_bin = cv2.erode(
        ~img_final_bin,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2
    )

    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    end = time.time()
    print(fr'Mark Regions Finished - File {file_name} === {end - start}')
    extractContours(contours, img, main_path, image_name)


def extractTxtFromImage(image_path_list, image_name):
    print(fr'Extract txt From Image Started - File {image_name}')
    start = time.time()

    transc_data = []
    invoice_city = 'SP' if 'SP' in image_name else 'RJ' if 'RJ' in image_name else None
    if invoice_city is not None:
        for path in image_path_list:
            img_roi_ocr = cv2.imread(path, 0)
            if img_roi_ocr is not None:
                config = r'-l eng+por --dpi 150'
                result = pytesseract.image_to_string(img_roi_ocr, config=config)
                transc_data.append(result)

        end = time.time()
        print(fr'Extract txt From Image Finished - File {image_name} === {end - start}')
        if invoice_city == 'RJ' and len(transc_data) > 0:
            normalizeRJData(transc_data, image_name)
        elif invoice_city == 'SP' and len(transc_data) > 0:
            normalizeSPData(transc_data, image_name)


def normalizeRJData(raw_data, image_name):
    print(fr'Normalize RJ Data Started - File {image_name}')
    start = time.time()

    invoice_info = {}
    invoice_raw_info = []

    for idx, value in enumerate(raw_data):
        aux = re.sub(r'\n', ' ', value)
        aux = re.sub(r'\s\s+', ' ', aux)

        invoice_info['invoice_city'] = {'value': 'RJ'}

        if len(aux) < 5:
            continue

        invoice_raw_info.append(aux)

        if fuzz.partial_ratio(aux.lower(), "numero da nota") > 90:
            key_value = re.search(r'\d+', aux)
            invoice_info['invoice_num'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "data e hora de emissão") > 90 or fuzz.partial_ratio(aux.lower(), "data e hora de emissao") > 90:
            key_value = re.search(r'(\d{2})\/(\d{2})\/(\d{4})\s(\d{2}):(\d{2}):(\d{2})', aux)
            invoice_info['invoice_creation'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "código de verificação") > 90 or fuzz.partial_ratio(aux.lower(), "codigo de verificacao") > 90:
            key_value = re.search(r'(?<=\s)([\w\|\d]+)-([\w\|\d]+)', aux)
            invoice_info['invoice_verif_cod'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "prestador de serviços") > 90 or fuzz.partial_ratio(aux.lower(), "prestador de servicos") > 90:
            doc_number_value = re.search(r'(?<=CPF\/CNPJ:\s)(([0-9]{3}\.?[0-9]{3}\.?[0-9]{3}\-?[0-9]{2}|[0-9]{2}\.?[0-9]{3}\.?[0-9]{3}\/?[0-9]{4}\s?\-?\s?[0-9]{2}))(?=\s)', aux)
            city_number_value = re.search(r'(?<=Inscrição Municipal:\s)(.*?)(?=\s)', aux)
            state_number_value = re.search(r'(?<=Inscrição Estadual:\s)(.*?)(?=\s)', aux)
            main_name_value = re.search(r'(?<=Nome\/Raz[a\|ã]o Social:\s)(.*?)(?=\sNome Fantasia)', aux)
            sec_name_value = re.search(r'(?<=Nome Fantasia:\s)(.*?)(?=\sTel)', aux)
            phone_value = re.search(r'(?<=Tel\.:\s)(\(?)(\d{2})(\)?)(\s?)(-?)(\d{4})(\s|-?)(\d{4})(?=\s)()', aux)
            address_value = re.search(r'(?<=Endereço:\s)(.*?)(?=\sCEP)', aux)
            zip_code_value = re.search(r'(?<=CEP:\s)(\d{5}-\d{3})', aux)
            city_value = re.search(r'(?<=Município:\s)(.*?)(?=\sUF)', aux)
            state_value = re.search(r'(?<=UF:\s)(.*?)(?=\s)', aux)
            email_value = re.search(r'(?<=E-mail:\s)(.*?)(?=\s)', aux)

            invoice_info['invoice_provider'] = {
                'doc_number': {'value': doc_number_value.group() if doc_number_value is not None else None},
                'city_number': {'value': city_number_value.group() if city_number_value is not None else None},
                'state_number': {'value': state_number_value.group() if state_number_value is not None else None},
                'main_name': {'value': main_name_value.group() if main_name_value is not None else None},
                'sec_name': {'value': sec_name_value.group() if sec_name_value is not None else None},
                'phone': {'value': phone_value.group() if phone_value is not None else None},
                'address': {'value': address_value.group() if address_value is not None else None},
                'zip_code': {'value': zip_code_value.group() if zip_code_value is not None else None},
                'city': {'value': city_value.group() if city_value is not None else None},
                'state': {'value': state_value.group() if state_value is not None else None},
                'email': {'value': email_value.group() if email_value is not None else None},
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "tomador de serviços") > 90 or fuzz.partial_ratio(aux.lower(), "tomador de servicos") > 90:
            doc_number_key_value = re.search(r'(?<=CPF\/CNPJ:\s)(([0-9]{3}\.?[0-9]{3}\.?[0-9]{3}\-?[0-9]{2}|[0-9]{2}\.?[0-9]{3}\.?[0-9]{3}\/?[0-9]{4}\s?\-?\s?[0-9]{2}))(?=\s)', aux)
            city_number_key_value = re.search(r'(?<=\sMunicipal:\s)(.*?)(?=\s)', aux)
            state_number_key_value = re.search(r'(?<=Inscriç[a|ã]o Estadual:\s)(.*?)(?=\s)', aux)
            main_name_key_value = re.search(rf'(?<=Nome\/Raz[a|ã]o Social:\s)(.*?)(?=\sEndere[c|ç]o:)', aux)
            address_key_value = re.search(r'(?<=Endere[c|ç|g]o:\s)(.*?)(?=\sCEP)', aux)
            zip_code_key_value = re.search(r'(?<=CEP:\s)(\d{5}-\d{3})', aux)
            city_key_value = re.search(r'(?<=Munic[í|i]pio:\s)(.*?)(?=\sUF)', aux)
            state_key_value = re.search(r'(?<=UF:\s)(.*?)(?=\s)', aux)
            email_key_value = re.search(r'(?<=E-mail:\s)(.*?)(?=\s)', aux)

            invoice_info['invoice_client'] = {
                'doc_number': {'value': doc_number_key_value.group() if doc_number_key_value is not None else None},
                'city_number': {'value': city_number_key_value.group() if city_number_key_value is not None else None},
                'state_number': {'value': state_number_key_value.group() if state_number_key_value is not None else None},
                'main_name': {'value': main_name_key_value.group() if main_name_key_value is not None else None},
                'address': {'value': address_key_value.group() if address_key_value is not None else None},
                'zip_code': {'value': zip_code_key_value.group() if zip_code_key_value is not None else None},
                'city': {'value': city_key_value.group() if city_key_value is not None else None},
                'state': {'value': state_key_value.group() if state_key_value is not None else None},
                'email': {'value': email_key_value.group() if email_key_value is not None else None},
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "discriminação dos serviços") > 90 or fuzz.partial_ratio(aux.lower(), "discriminacao dos servicos") > 90:
            order_number = None
            order_number_paterns = [r'[N|n][º] [D|d][O|o] [P|p][E|e][D|d][I|i][D|d][O|o]',
                                    r'[O|o][R|r][D|d][E|e][M|m] [D|d][E|e] [C|c][O|o][M|m][P|p][R|r][A|a]',
                                    r'[P|p][E|e][D|d][I|i][D|d][O|o] [D|d][E|e] [C|c][O|o][M|m][P|p][R|r][A|a]',
                                    r'[P|p][E|e][D|d][I|i][D|d][O|o] [D|d][O|o] [P|p][E|e][D|d][I|i][D|d][O|o]',
                                    r'[P|p][E|e][D|d][I|i][D|d][O|o] [O|o][C|c]',
                                    r'[P|p][E|e][D|d][I|i][D|d][O|o]']

            for op in order_number_paterns:
                if order_number is None:
                    order_number = re.search(fr'(?<={op}:\s)(.*?)(?=\s)', aux)
                    if order_number is not None:
                        order_number = order_number.group()
                        break

            invoice_info['invoice_description'] = {
                'order_number': order_number,
                'raw': aux
            }

            olv_pattern = r'[V|v][A|a][L|l][O|o][R|r] [L|l][I|i][Q|q][U|u][I|Í|i|í][D|d][O|o]'
            key_value_aux = re.search(fr'(?<={olv_pattern}).*', aux)
            key_value_aux = key_value_aux.group() if key_value_aux is not None else ''
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}(?=\s)', key_value_aux)

            invoice_info['invoice_value_liq'] = {
                'value': key_value.group() if key_value is not None else None,
            }

        if fuzz.partial_ratio(aux.lower(), "valor da nota") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_value_raw'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "serviço prestado") > 90 or fuzz.partial_ratio(aux.lower(), "servico prestado") > 90:
            key_value = re.search(r'(?<=Prestado\s)(\d+\.\d+\.\d).*', aux)
            invoice_info['invoice_service'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "deduções") > 90 or fuzz.partial_ratio(aux.lower(), "deducoes") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_deduc'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "desconto incond.") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_discount'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "base de cálculo") > 90 or fuzz.partial_ratio(aux.lower(), "base de calculo") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_calc_base_tax'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "aliquota") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}(\s?%?)', aux)
            invoice_info['invoice_aliq_tax'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "valor de iss") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_iss_tax'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "crédito p/ iptu") > 90 or fuzz.partial_ratio(aux.lower(), "credito p/ iptu") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_iptu_credit'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

    end = time.time()
    print(fr'Normalize RJ Data Finished - File {image_name} === {end - start}')
    saveProcessResult(invoice_raw_info, invoice_info, image_name)
    # return invoice_info


def normalizeSPData(raw_data, image_name):
    print(fr'Normalize SP Data Started - File {image_name}')
    start = time.time()

    invoice_info = {}
    invoice_raw_info = []

    for idx, value in enumerate(raw_data):
        aux = re.sub(r'\n', ' ', value)
        aux = re.sub(r'\s\s+', ' ', aux)

        invoice_info['invoice_city'] = {'value': 'SP'}

        if len(aux) < 5:
            continue

        invoice_raw_info.append(aux)

        if fuzz.partial_ratio(aux.lower(), "numero da nota") > 90:
            key_value = re.search(r'\d+', aux)
            invoice_info['invoice_num'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "data e hora de emissão") > 90 or fuzz.partial_ratio(aux.lower(), "data e hora de emissao") > 90:
            key_value = re.search(r'(\d{2})\/(\d{2})\/(\d{4})\s(\d{2}):(\d{2}):(\d{2})', aux)
            invoice_info['invoice_creation'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "código de verificação") > 90 or fuzz.partial_ratio(aux.lower(), "codigo de verificacao") > 90:
            key_value = re.search(r'(?<=\s)([\w\|\d]+)-([\w\|\d]+)', aux)
            invoice_info['invoice_verif_cod'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "prestador de serviços") > 90 or fuzz.partial_ratio(aux.lower(), "prestador de servicos") > 90:
            doc_number_value = re.search(r'(?<=CPF\/CNPJ:\s)(([0-9]{3}\.?[0-9]{3}\.?[0-9]{3}\-?[0-9]{2}|[0-9]{2}\.?[0-9]{3}\.?[0-9]{3}\/?[0-9]{4}\s?\-?\s?[0-9]{2}))(?=\s)', aux)
            city_number_value = re.search(r'(?<=Inscri[c|ç][a|ã]o Municipal:\s)(.*?)(?=\s)', aux)
            main_name_value = re.findall(r'(?:Nome\/[A-zÀ-ÿ]+?\sSocial.\s)(.*)(?=\sEndere[c|ç]o)', aux)
            address_value = re.search(r'(?<=Endere[c|ç]o:\s)(.*?)(?=\sCEP)', aux)
            zip_code_value = re.search(r'(?<=CEP:\s)(\d{5}-\d{3})', aux)
            city_value = re.search(r'(?<=Munic[i|í]pio:\s)(.*?)(?=\sUF)', aux)
            state_value = re.search(r'(?<=UF:\s)(.*?)(?=\s)', aux)

            invoice_info['invoice_provider'] = {
                'doc_number': {'value': doc_number_value.group() if doc_number_value is not None else None},
                'city_number': {'value': city_number_value.group() if city_number_value is not None else None},
                'main_name': {'value': main_name_value[0] if len(main_name_value) > 0 else None},
                'address': {'value': address_value.group() if address_value is not None else None},
                'zip_code': {'value': zip_code_value.group() if zip_code_value is not None else None},
                'city': {'value': city_value.group() if city_value is not None else None},
                'state': {'value': state_value.group() if state_value is not None else None},
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "tomador de serviços") > 90 or fuzz.partial_ratio(aux.lower(), "tomador de servicos") > 90:
            main_name_key_value = re.findall(r'(?:Nome\/[A-zÀ-ÿ]+?\sSocial.\s)(.*)(?=\sCPF\/CNPJ.)', aux)
            doc_number_key_value = re.search(r'(?<=CPF\/CNPJ:\s)(([0-9]{3}\.?[0-9]{3}\.?[0-9]{3}\-?[0-9]{2}|[0-9]{2}\.?[0-9]{3}\.?[0-9]{3}\/?[0-9]{4}\s?\-?\s?[0-9]{2}))(?=\s)', aux)
            city_number_key_value = re.search(r'(?<=\sMunicipal:\s)(.*?)(?=\s)', aux)
            address_key_value = re.search(r'(?<=Endere[c|ç|g]o:\s)(.*?)(?=\sCEP)', aux)
            zip_code_key_value = re.search(r'(?<=CEP:\s)(\d{5}-\d{3})', aux)
            city_key_value = re.search(r'(?<=Munic[í|i]pio:\s)(.*?)(?=\sUF)', aux)
            state_key_value = re.search(r'(?<=UF:\s)(.*?)(?=\s)', aux)
            email_key_value = re.search(r'(?<=E-mail:\s)(.*?)(?=\s)', aux)

            invoice_info['invoice_client'] = {
                'main_name': {'value': main_name_key_value[0] if len(main_name_key_value) > 0 else None},
                'doc_number': {'value': doc_number_key_value.group() if doc_number_key_value is not None else None},
                'city_number': {'value': city_number_key_value.group() if city_number_key_value is not None else None},
                'address': {'value': address_key_value.group() if address_key_value is not None else None},
                'zip_code': {'value': zip_code_key_value.group() if zip_code_key_value is not None else None},
                'city': {'value': city_key_value.group() if city_key_value is not None else None},
                'state': {'value': state_key_value.group() if state_key_value is not None else None},
                'email': {'value': email_key_value.group() if email_key_value is not None else None},
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "discriminação dos serviços") > 90 or fuzz.partial_ratio(aux.lower(), "discriminacao dos servicos") > 90:
            order_number = None
            order_number_paterns = [r'[N|n][º] [D|d][O|o] [P|p][E|e][D|d][I|i][D|d][O|o]',
                                    r'[O|o][R|r][D|d][E|e][M|m] [D|d][E|e] [C|c][O|o][M|m][P|p][R|r][A|a]',
                                    r'[P|p][E|e][D|d][I|i][D|d][O|o] [D|d][E|e] [C|c][O|o][M|m][P|p][R|r][A|a]',
                                    r'[P|p][E|e][D|d][I|i][D|d][O|o] [D|d][O|o] [P|p][E|e][D|d][I|i][D|d][O|o]',
                                    r'[P|p][E|e][D|d][I|i][D|d][O|o] [O|o][C|c]',
                                    r'[P|p][E|e][D|d][I|i][D|d][O|o]']

            for op in order_number_paterns:
                if order_number is None:
                    order_number = re.search(fr'(?<={op}:\s)(.*?)(?=\s)', aux)
                    if order_number is not None:
                        order_number = order_number.group()
                        break

            invoice_info['invoice_description'] = {
                'order_number': order_number,
                'raw': aux
            }

            olv_pattern = r'[V|v][A|a][L|l][O|o][R|r] [L|l][I|i][Q|q][U|u][I|Í|i|í][D|d][O|o]'
            key_value_aux = re.search(fr'(?<={olv_pattern}).*', aux)
            key_value_aux = key_value_aux.group() if key_value_aux is not None else ''
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}(?=\s)', key_value_aux)

            invoice_info['invoice_value_liq'] = {
                'value': key_value.group() if key_value is not None else None,
            }

        if fuzz.partial_ratio(aux.lower(), "valor total do serviço") > 90 or fuzz.partial_ratio(aux.lower(), "valor total do servico") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_value_raw'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "inss") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_inss'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "irrf") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_irrf'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "csll") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_csll'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "cofins") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_cofins'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "pis/pasep") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_pis-pasep'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "código do serviço") > 90 or fuzz.partial_ratio(aux.lower(), "código do servico") > 90:
            key_value = re.search(r'(?<=\s)(\d+).*', aux)
            invoice_info['invoice_service'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "valor total das deduções") > 90 or fuzz.partial_ratio(aux.lower(), "valor total das deducoes") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_deductions'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "base de cálculo") > 90 or fuzz.partial_ratio(aux.lower(), "base de calculo") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_calc_base'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "aliquota") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}(\s?%?)', aux)
            invoice_info['invoice_tax_aliq'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "valor de iss") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_iss'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "crédito (r$)") > 90 or fuzz.partial_ratio(aux.lower(), "credito (r$)") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_credit'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "município da prestação do serviço") > 90 or fuzz.partial_ratio(aux.lower(), "municipio da prestacao do servico") > 90:
            key_value = re.search(r'(?<= do Servi[c|ç]o\s).*', aux)
            invoice_info['invoice_tax_service_city'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "número inscrição da obra") > 90 or fuzz.partial_ratio(aux.lower(), "numero inscricao da obra") > 90:
            key_value = re.search(r'(?<=da Obra\s).*', aux)
            invoice_info['invoice_tax_work_num'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

        if fuzz.partial_ratio(aux.lower(), "valor aproximado dos tributos / fonte") > 90:
            key_value = re.search(r'(0|[1-9]\d{0,2}(\.\d{3})*),\d{2}', aux)
            invoice_info['invoice_tax_tribute_value'] = {
                'value': key_value.group() if key_value is not None else None,
                'raw': aux
            }

    end = time.time()
    print(fr'Normalize SP Data Finished - File {image_name} === {end - start}')
    saveProcessResult(invoice_raw_info, invoice_info, image_name)
    # return invoice_info


def saveProcessResult(result_raw_obj, result_obj, image_name):
    print(fr'Save Result Started - File {image_name}')
    start = time.time()

    json_string = json.dumps(result_obj, indent=2, sort_keys=True, ensure_ascii=False)
    json_file = open(fr'./input/{image_name}.json', 'w')
    json_file.write(json_string)
    json_file.close()

    text_file = open(fr'./input/{image_name}.txt', 'w')
    for el in result_raw_obj:
        text_file.write(el + '\n')
    text_file.close()

    shutil.rmtree(fr'./processing/{image_name}')

    end = time.time()
    print(fr'Save Result Finished - File {image_name} === {end - start}')


if __name__ == "__main__":
    while True:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(fr'Cicle started - {dt_string}')

        new_files = os.listdir('./input')
        for file in new_files:
            process_running = os.listdir('./processing')
            file_name = os.path.splitext(file)[0]
            try:
                if fr'{file_name}.json' in new_files:
                    continue
                elif fr'{file_name}' not in process_running:
                    os.mkdir(fr'./processing/{file_name}')
                process_running = os.listdir('./processing')
                for proc in process_running:
                    proc_dir = os.listdir(fr'./processing/{proc}') if proc != '.DS_Store' else [False]
                    if len(proc_dir) == 0:
                        convertPDF2Image(proc)
            except Exception as err:
                print(fr'Error when running {file} file transcript. - {err}')
        time.sleep(30)
