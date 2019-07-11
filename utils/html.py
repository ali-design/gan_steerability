from os import listdir
from os.path import isfile, join


def make_html(images_dir):
    file_names = [f for f in listdir(images_dir) if join(images_dir, f).endswith('.png')]

    try:
        fid = open(join(images_dir, "index.html"), 'w', encoding = 'utf-8')

        fid.write('<table style="text-align:center;">')

        fid.write('<tr><td>Image #</td><td>Output</td></tr>')

        for i in range(len(file_names)):
            print(file_names[i])
            fid.write('<tr>')
            fid.write('<td>' + file_names[i] + '</td>')
            fid.write('<td><a href="' + file_names[i] + '"><img src="' +
                      file_names[i] + '" width="600"/></a></td>')
            fid.write('</tr>')

        fid.write('</table>')

    finally:
        fid.close()
