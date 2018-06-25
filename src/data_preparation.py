def crop_images(images_path, image_size, output_folder='dataset/Cropped'):
    total_folders = sum(1 for e in os.scandir(images_path) if e.is_dir())
    for entry in tqdm_notebook(os.scandir(images_path), total=total_folders, unit="folders"):
        if not entry.name.startswith('.') and entry.is_dir():
            crop_folder_images(entry.path, image_size)

def crop_folder_images(folder_path, image_size, output_folder='dataset/Cropped'):
    for entry in os.scandir(folder_path):
        if not entry.name.startswith('.') and entry.is_file():
            crop_image(entry.path, image_size)
                    
def crop_image(image_path, image_size, output_folder='dataset/Cropped'):
    try:
        image_data = imageio.imread(image_path, pilmode='RGB') #RGB mode to cut-out alpha layer, if exists

        annon_xml = minidom.parse(image_path.replace("Images", "Annotation")[:-4])
        
        xmin = int(annon_xml.getElementsByTagName('xmin')[0].firstChild.nodeValue)
        ymin = int(annon_xml.getElementsByTagName('ymin')[0].firstChild.nodeValue)
        xmax = int(annon_xml.getElementsByTagName('xmax')[0].firstChild.nodeValue)
        ymax = int(annon_xml.getElementsByTagName('ymax')[0].firstChild.nodeValue)

        new_image_data = image_data[ymin:ymax,xmin:xmax,:]
        new_image_data = resize(new_image_data, image_size, preserve_range=True).astype('uint8')
        
        output_folder_path = os.path.join(output_folder, image_path.split('/')[-2])
        image_name = image_path.split('/')[-1]
        
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        
        imageio.imwrite(
            os.path.join(output_folder_path, image_name), 
            new_image_data
        )
        
    except IOError as e:
        print('Could not read: {}: {} - skipping.'.format(image_path, e))