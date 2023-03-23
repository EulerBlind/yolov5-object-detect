import shutil
from collections import Counter
from pathlib import Path, PosixPath
from typing import Dict
from xml.dom.minidom import Document
import numpy as np
import cv2
from imagededup.methods import PHash
import xml.etree.ElementTree as ET


def _cvt_xml2yolo(path_xml: Path, classes: dict):
    path_label = Path(path_xml.parent.parent, "labels", f"{path_xml.stem}.txt")
    if not path_label.exists():
        path_label.touch()
    with open(str(path_label.resolve()), 'w') as label_file:
        with open(str(path_xml.resolve()), "r", encoding='UTF-8') as xml_file:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            size_width = int(size.find('width').text)
            size_height = int(size.find('height').text)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes[cls]
                xmlbox = obj.find('bndbox')
                b = [float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text)]

                if size_width == 0 or size_height == 0 or b[0] == b[1] or b[2] == b[3]:
                    print("不合理的图不再给labels  ", path_xml.stem)
                    label_file.close()
                    path_label.unlink()
                    return 1
                # 标注越界修正
                b[0] = max(0, b[0])
                b[1] = min(size_width, b[1])
                b[2] = max(0, b[2])
                b[3] = min(size_height, b[3])
                txt_data = [round(((b[0] + b[1]) / 2.0 - 1) / size_width, 6),
                            round(((b[2] + b[3]) / 2.0 - 1) / size_height, 6),
                            round((b[1] - b[0]) / size_width, 6),
                            round((b[3] - b[2]) / size_height, 6)]
                if txt_data[0] < 0 or txt_data[1] < 0 or txt_data[2] < 0 or txt_data[3] < 0:
                    print("不合理的图不再给labels  ", path_xml.stem)
                    label_file.close()
                    path_label.unlink()
                    return 1
                label_file.write(str(cls_id) + " " + " ".join([str(a) for a in txt_data]) + '\n')
        return 0


def _cvt_yolo2xml(path_label: Path, img_map: Dict[str, Path], class_dict: dict):
    """
     此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    :param path_label: yolo格式txt标注文件路径
    :param img_map: 图片名称与路径的映射
    :param class_dict: 类别名称与类别id的映射 {'0': "invoice"}
    :return:
    """
    path_xml = Path(path_label.parent.parent, "xml", f"{path_label.stem}.xml")
    if not path_xml.exists():
        path_xml.touch()
    xmlBuilder = Document()
    annotation = xmlBuilder.createElement("annotation")
    xmlBuilder.appendChild(annotation)
    with open(str(path_label.resolve()), "r", encoding='UTF-8') as label_file:
        txtList = label_file.readlines()
        path_img = img_map.get(path_label.stem)
        if path_img is None:
            return
        img = cv2.imdecode(np.fromfile(str(path_img.resolve()), dtype=np.uint8), 1)
        shape_height, shape_width, shape_depth = img.shape
        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(path_img.name)
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(shape_width))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(shape_height))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(shape_depth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束
        annotation.appendChild(size)  # size标签结束

        for j in txtList:
            oneline = j.strip().split(" ")
            if len(oneline) > 4:
                object = xmlBuilder.createElement("object")  # object 标签
                picname = xmlBuilder.createElement("name")  # name标签
                namecontent = xmlBuilder.createTextNode(class_dict[oneline[0]])
                picname.appendChild(namecontent)
                object.appendChild(picname)  # name标签结束

                pose = xmlBuilder.createElement("pose")  # pose标签
                posecontent = xmlBuilder.createTextNode("Unspecified")
                pose.appendChild(posecontent)
                object.appendChild(pose)  # pose标签结束

                truncated = xmlBuilder.createElement("truncated")  # truncated标签
                truncatedContent = xmlBuilder.createTextNode("0")
                truncated.appendChild(truncatedContent)
                object.appendChild(truncated)  # truncated标签结束

                difficult = xmlBuilder.createElement("difficult")  # difficult标签
                difficultcontent = xmlBuilder.createTextNode("0")
                difficult.appendChild(difficultcontent)
                object.appendChild(difficult)  # difficult标签结束

                bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
                xmin = xmlBuilder.createElement("xmin")  # xmin标签
                mathData = int(((float(oneline[1])) * shape_width + 1) - (float(oneline[3])) * 0.5 * shape_width)
                xminContent = xmlBuilder.createTextNode(str(mathData))
                xmin.appendChild(xminContent)
                bndbox.appendChild(xmin)  # xmin标签结束

                ymin = xmlBuilder.createElement("ymin")  # ymin标签
                mathData = int(((float(oneline[2])) * shape_height + 1) - (float(oneline[4])) * 0.5 * shape_height)
                yminContent = xmlBuilder.createTextNode(str(mathData))
                ymin.appendChild(yminContent)
                bndbox.appendChild(ymin)  # ymin标签结束

                xmax = xmlBuilder.createElement("xmax")  # xmax标签
                mathData = int(((float(oneline[1])) * shape_width + 1) + (float(oneline[3])) * 0.5 * shape_width)
                xmaxContent = xmlBuilder.createTextNode(str(mathData))
                xmax.appendChild(xmaxContent)
                bndbox.appendChild(xmax)  # xmax标签结束

                ymax = xmlBuilder.createElement("ymax")  # ymax标签
                mathData = int(((float(oneline[2])) * shape_height + 1) + (float(oneline[4])) * 0.5 * shape_height)
                ymaxContent = xmlBuilder.createTextNode(str(mathData))
                ymax.appendChild(ymaxContent)
                bndbox.appendChild(ymax)  # ymax标签结束
                object.appendChild(bndbox)  # bndbox标签结束
                annotation.appendChild(object)  # object标签结束
            with open(str(path_xml.resolve()), "w", encoding='UTF-8') as xml_file:
                xmlBuilder.writexml(xml_file, indent='\t', newl='\n', addindent='\t', encoding='utf-8')


def trans_image_2_jpg(base_path: str):
    """
    将图片转换成jpg格式
    :param base_path: 源文件路径
    :return:
    """
    src_path = Path(base_path, "images")
    out_path = Path(base_path, "images_out")
    if not out_path.exists():
        out_path.mkdir()
    if not out_path.exists():
        out_path.mkdir()
    backup_path = Path(base_path, "images_bak")
    if backup_path.exists():
        print(f"{backup_path.absolute()}文件夹已存在，请先删除")
        return
    images: [PosixPath] = [*src_path.glob("*.jpg"), *src_path.glob("*.png"), *src_path.glob("*.jpeg")]
    for image in images:
        try:
            img = cv2.imread(str(image.resolve()))
            cv2.imwrite(str(Path(out_path, image.stem + ".jpg").resolve()), img)
        except Exception as e:
            print(f"读取文件失败 {image.resolve()} {e}")
    src_path.rename(backup_path)
    out_path.rename(Path(base_path, "images"))


def clean_same_images(base_path: str):
    """
    清理重复的图片
    :param base_path: 根路径
    :return:
    """
    path_img = Path(base_path, "images")
    path_xml = Path(base_path, "xml")
    path_img = Path(base_path, "images")
    path_label = Path(base_path, "labels")
    # image_files: [PosixPath] = [*path_img.glob("*.jpg"), *path_img.glob("*.png"), *path_img.glob("*.jpeg")]
    label_map = {label.stem: label for label in path_label.glob("*.txt")}
    xml_map = {label.stem: label for label in path_xml.glob("*.xml")}
    phasher = PHash()
    encodings = phasher.encode_images(image_dir=str(path_img.absolute()))
    exits_encoding = set()
    for file, encoding in encodings.items():
        if encoding in exits_encoding:
            exist_img = Path(path_img, file)
            print(f"删除重复图片 {exist_img.resolve()}")
            exist_img.unlink()
            label_file = label_map.get(exist_img.stem)
            if label_file:
                print(f"删除重复图片label {label_file.resolve()}")
                label_file.unlink()
            xml_file = xml_map.get(exist_img.stem)
            if xml_file:
                print(f"删除重复图片xml {xml_file.resolve()}")
                xml_file.unlink()
        exits_encoding.add(encoding)


def rename_label_file(base_path: str, new_name: str):
    """
    重命名数据数
    如果存在xml与label 命名为 {new_name}_xml_label_idx
    如果存在xml 命名为 {new_name}_xml_idx
    如果存在label 命名为 {new_name}_label_idx
    如果都不存 重命名为 {new_name}_img_idx
    :param base_path: 数据跟路径
    :return:
    """
    path_xml = Path(base_path, "xml")
    path_img = Path(base_path, "images")
    path_label = Path(base_path, "labels")
    image_files: [PosixPath] = [*path_img.glob("*.jpg"), *path_img.glob("*.png"), *path_img.glob("*.jpeg")]
    image_file_size_dict = {file: file.stat().st_size for file in image_files}
    image_files = [file[0] for file in sorted(image_file_size_dict.items(), key=lambda x: x[1], reverse=True)]
    label_map = {label.stem: label for label in path_label.glob("*.txt")}
    xml_map = {label.stem: label for label in path_xml.glob("*.xml")}
    img_idx = img_xml_label_idx = img_xml_idx = img_label_idx = 0
    for image in image_files:
        if image.stem in xml_map and image.stem in label_map:
            img_xml_label_idx += 1
            re_path = f"{new_name}_xml_label_{img_xml_label_idx}"
            image.rename(Path(path_img, f"{re_path}{image.suffix}"))
            xml_map.get(image.stem).rename(Path(path_xml, f"{re_path}.xml"))
            label_map.get(image.stem).rename(Path(path_label, f"{re_path}.txt"))
        elif image.stem in xml_map and image.stem not in label_map:
            img_xml_idx += 1
            re_path = f"{new_name}_xml_{img_xml_idx}"
            image.rename(Path(path_img, f"{re_path}{image.suffix}"))
            xml_map.get(image.stem).rename(Path(path_xml, f"{re_path}.xml"))
        elif image.stem not in xml_map and image.stem in label_map:
            img_label_idx += 1
            re_path = f"{new_name}_label_{img_label_idx}"
            image.rename(Path(path_img, f"{re_path}{image.suffix}"))
            label_map.get(image.stem).rename(Path(path_label, f"{re_path}.txt"))
        else:
            img_idx += 1
            re_path = f"{new_name}_img_{img_idx}"
            image.rename(Path(path_img, f"{re_path}{image.suffix}"))


def clean_by_xml(base_path: str):
    """
    如果xml文件不存在，清除 img，label
    :param base_path:
    :return:
    """
    sure = input("确定清除xml不存在的图片和label文件？(y/n)")
    if sure == "n" or sure == "N":
        return
    path_xml = Path(base_path, "xml")
    path_img = Path(base_path, "images")
    path_label = Path(base_path, "labels")

    xml_files = path_xml.glob("*.xml")
    xml_stems = [xml.stem for xml in xml_files]
    label_files = path_label.glob("*.txt")
    image_files: [PosixPath] = [*path_img.glob("*.jpg"), *path_img.glob("*.png"), *path_img.glob("*.jpeg")]
    [file.unlink() for file in label_files if file.stem not in xml_stems]
    [file.unlink() for file in image_files if file.stem not in xml_stems]


def clean_by_label(base_path: str):
    """
    如果label文件不存在，清除 img，xml
    :param base_path:
    :return:
    """
    sure = input("确定清除label不存在的图片和xml文件？(y/n)")
    if sure == "n" or sure == "N":
        return
    path_xml = Path(base_path, "xml")
    path_img = Path(base_path, "images")
    path_label = Path(base_path, "labels")

    xml_files = path_xml.glob("*.xml")
    label_files = path_label.glob("*.txt")
    label_stem = [label.stem for label in label_files]
    image_files: [PosixPath] = [*path_img.glob("*.jpg"), *path_img.glob("*.png"), *path_img.glob("*.jpeg")]
    [file.unlink() for file in xml_files if file.stem not in label_stem]
    [file.unlink() for file in image_files if file.stem not in label_stem]


def clean_xml_and_label_if_img_not_exist(base_path: str):
    """
    如果图片不存在，清除xml和label
    :param base_path:
    :return:
    """
    sure = input("确定清除不存在图片的xml与label吗?[y/n]")
    if sure == "n" or sure == "N":
        return
    path_img = Path(base_path, "images")
    path_xml = Path(base_path, "xml")
    path_label = Path(base_path, "labels")

    label_files = path_label.glob("*.txt")
    xml_files = path_xml.glob("*.xml")
    image_files: [PosixPath] = [*path_img.glob("*.jpg"), *path_img.glob("*.png"), *path_img.glob("*.jpeg")]
    image_stems = [file.stem for file in image_files]
    [file.unlink() for file in xml_files if file.stem not in image_stems]
    [file.unlink() for file in label_files if file.stem not in image_stems]


def gen_negative_label(base_path: str):
    """
    生成空标签(反例标签)
    :param base_path: 标签路径
    :return:
    """
    path_img = Path(base_path, "images")
    path_xml = Path(base_path, "xml")
    path_label = Path(base_path, "labels")
    xml_files = path_xml.glob("*.xml")
    xml_file_names = [file.stem for file in xml_files]
    image_paths: [PosixPath] = [*path_img.glob("*.jpg"), *path_img.glob("*.png"), *path_img.glob("*.jpeg")]
    need_create_labels = [Path(path_label, image.stem + ".txt") for image in image_paths if
                          image.stem not in xml_file_names]
    print(f"创建空标签文件：\n{need_create_labels}")
    [file.touch() for file in need_create_labels]


def cvt_voc2yolo(base_path: str, class_dict: dict):
    path_xml = Path(base_path, "xml")
    path_label = Path(base_path, "labels")
    if not path_label.exists():
        path_label.mkdir()
    file_xml = path_xml.glob("*.xml")
    [_cvt_xml2yolo(file, class_dict) for file in file_xml]


def cvt_yolo2voc(base_path: str, class_dict: dict):
    # path_xml = Path(base_path, "xml")
    path_img = Path(base_path, "images")
    path_label = Path(base_path, "labels")
    image_paths: [PosixPath] = [*path_img.glob("*.jpg"), *path_img.glob("*.png"), *path_img.glob("*.jpeg")]
    image_stems_map = {file.stem: file for file in image_paths}
    label_files = path_label.glob("*.txt")
    [_cvt_yolo2xml(file, image_stems_map, class_dict) for file in label_files]


def select_file_2_mark(base_path: str, select_num: int = 100):
    path_img = Path(base_path, "images")
    path_xml = Path(base_path, "xml")
    path_label = Path(base_path, "labels")
    xml_map = {file.stem: file for file in path_xml.glob("*.xml")}
    label_map = {file.stem: file for file in path_label.glob("*.txt")}
    image_paths: [PosixPath] = [*path_img.glob("*.jpg"), *path_img.glob("*.png"), *path_img.glob("*.jpeg")]
    new_path = None
    for i in range(10):
        new_path = Path(Path(base_path).parent, f"{base_path.split('/')[-1]}{i + 1}")
        if new_path.exists():
            continue
        new_path.mkdir()
        break
    if new_path is None:
        print("多个数据文件夹存在，请及时清理后使用")
        return
    print(f"新文件夹路径：{new_path}")
    new_path_img = Path(new_path, "images")
    if not new_path_img.exists():
        new_path_img.mkdir()
    new_path_xml = Path(new_path, "xml")
    if not new_path_xml.exists():
        new_path_xml.mkdir()
    new_path_label = Path(new_path, "labels")
    if not new_path_label.exists():
        new_path_label.mkdir()
    for idx, image_path in enumerate(image_paths):
        if idx >= select_num:
            return
        new_image_path = Path(new_path_img, image_path.name)
        new_image_path.touch()
        shutil.copy(image_path, new_image_path)
        old_xml_path = xml_map.get(image_path.stem)
        if old_xml_path:
            new_xml_path = Path(new_path_xml, image_path.stem + ".xml")
            new_xml_path.touch()
            shutil.copy(old_xml_path, new_xml_path)
        old_label_path = label_map.get(image_path.stem)
        if old_label_path:
            new_label_path = Path(new_path_label, image_path.stem + ".txt")
            new_label_path.touch()
            shutil.copy(old_label_path, new_label_path)


def stander_dirs(base_path: str):
    """
    标准化目录结构
    :param base_path:
    :return:
    """
    base = Path(base_path)
    path_img = Path(base_path, "images")
    path_xml = Path(base_path, "xml")
    path_label = Path(base_path, "labels")
    if not path_img.exists():
        path_img.mkdir()
    if not path_xml.exists():
        path_xml.mkdir()
    if not path_label.exists():
        path_label.mkdir()
    image_paths: [PosixPath] = [*base.glob("*.jpg"), *base.glob("*.png"), *base.glob("*.jpeg")]
    xml_files = base.glob("*.xml")
    label_files = base.glob("*.txt")
    [shutil.move(str(path.absolute()), path_img) for path in image_paths]
    [shutil.move(str(path.absolute()), path_xml) for path in xml_files]
    [shutil.move(str(path.absolute()), path_label) for path in label_files]


def move_img_to_jpg(base_path: str, keyword: str):
    path_aim = Path(base_path, keyword)
    if not path_aim.exists():
        path_aim.mkdir()
    aim_dirs = Path(base_path).glob(f"{keyword}_*")
    # aim_dirs = [Path(base_path, "足球")]
    img_idx = 0
    for aim_dir in aim_dirs:
        image_paths: [PosixPath] = [*aim_dir.glob("*.jpg"), *aim_dir.glob("*.png"), *aim_dir.glob("*.jpeg")]
        for image in image_paths:
            img_idx += 1
            re_path = f"{keyword}_{img_idx}"
            print(f"rename {image.name} to {re_path}{image.suffix}")
            image.rename(Path(path_aim, f"{re_path}{image.suffix}"))
        aim_dir.rmdir()


if __name__ == '__main__':
    # base_path = "/Users/eulerblind/PycharmProjects/pytorch-learning/train"
    # base_path = "data/gym"
    base_path = "/Users/eulerblind/PycharmProjects/deeplearing-study/yolov5-object-dectect/dataset"
    # move_img_to_jpg(base_path, "football")
    # paths = ["./健身房_1", "./博物馆_1", "./手机_1", "./收据_1", "./明信片_2", "./机场_1","./游戏机_1",
    #         "./电脑_1","./相机_1" ,"./离婚证_2","./耳机_1","./酒吧_1","./门票_1","./音响_1","./餐厅_1","./餐厅_2"]
    # for path in paths:
    # stander_dirs(base_path)
    # trans_image_2_jpg(base_path)
    # clean_same_images(base_path, )
    # rename_label_file(base_path, "gym")

    # clean_by_xml(base_path)
    # clean_by_label(base_path)
    # clean_xml_and_label_if_img_not_exist(base_path)

    # select_file_2_mark(base_path, 250)
    # gen_negative_label(base_path)
    # cvt_yolo2voc(base_path, {"0": "cellphone"})
    # cvt_voc2yolo(base_path, {'football': 0})
