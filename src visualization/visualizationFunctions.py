import numpy as np
import itk
from itkwidgets import view
from ipywidgets import interactive, interact
import itkwidgets
import ipywebrtc as webrtc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

PixelTypeSS = itk.ctype('signed short')
PixelTypeUS = itk.ctype('unsigned short')
Dimension = 3

ImageTypeSS3 = itk.Image[PixelTypeSS, Dimension]
ImageTypeUS3 = itk.Image[PixelTypeUS, Dimension]

plt.style.use('dark_background')
plt.rcParams['animation.embed_limit'] = 2**128

# Util functions
def castImage(itkImage, inputType, outputType):
    castImageFilter = itk.CastImageFilter[inputType, outputType].New()
    castImageFilter.SetInput( itkImage)
    castImageFilter.Update()

    return castImageFilter.GetOutput()

def binaryShapeLabelMap(itkImage, inputForegroundValue=1):
    if type(itkImage) != ImageTypeUS3:
        itkImage = castImage(itkImage, type(itkImage), ImageTypeUS3)
    BinaryImageToShapeLabelMapFilterType = itk.BinaryImageToShapeLabelMapFilter[ImageTypeUS3, LabelMapTypeUL3]
    binaryImageToShapeLabelMapFilter = BinaryImageToShapeLabelMapFilterType.New()
    binaryImageToShapeLabelMapFilter.SetInput(itkImage)
    binaryImageToShapeLabelMapFilter.SetInputForegroundValue(inputForegroundValue)
    binaryImageToShapeLabelMapFilter.Update()

    return binaryImageToShapeLabelMapFilter.GetOutput()

def getBoundingBoxes(itkImage):
    #itkImage = castImage(itkImage, type(itkImage), ImageTypeUS3)
    labelMap = binaryShapeLabelMap(itkImage, inputForegroundValue=1)

    boundingBoxes = []
    for i in range(labelMap.GetNumberOfLabelObjects()):
        label = labelMap.GetNthLabelObject(i)
        boundingBoxes.append(itk.ImageRegion[3](label.GetBoundingBox()))

    return boundingBoxes

def extractSlice(itkImage, sliceNum, numberOfSlices=1):
    extractFilter = itk.ExtractImageFilter[type(itkImage), type(itkImage)].New()
    extractFilter.SetDirectionCollapseToSubmatrix()

    # set up the extraction region [one slice]
    inputRegion = itkImage.GetLargestPossibleRegion()
    size = inputRegion.GetSize()
    size[2] = numberOfSlices  # we extract along z direction
    start = inputRegion.GetIndex()
    sliceNumber = sliceNum
    start[2] = sliceNumber

    desiredRegion = inputRegion
    desiredRegion.SetSize(size)
    desiredRegion.SetIndex(start)

    extractFilter.SetExtractionRegion(desiredRegion)
    extractFilter.SetInput(itkImage)

    extractFilter.Update()
    
    return extractFilter.GetOutput()


# Visualization functions

def obtenerHistograma(numpyDicom, bins=80, xmin=None, xmax=None, ticks=100):
    """
    Returns a matplotlib histogram plot
    Retorna un plot de matplotlib con un histograma

    Parameters
    ----------
    numpyDicom : numpy array
        2D or 3D image in numpy format
        Imagen en formato numpy de dimensión 2D o 3D
    bins : int
        Rectangle number in wich the graph is divided
        Número de rectángulos en los que dividir el gráfico
    xmin : int
        Minimum axis x value
        Mínimo valor en el eje x
    xmax : int
        Máximum axis x value
        Máximo valor en el eje x
    ticks : int
        Distance between numbers shown in x axis
        Distancia entre los números que se muestran en el eje x

    Returns
    -------
    Matplotlib plot containing the histogram
    plot de matplotlib con el histograma
    """
    numpyDicomFlatten = numpyDicom.flatten()
    if xmin is not None:
        numpyDicomFlatten = numpyDicomFlatten[numpyDicomFlatten >= xmin]
        minX=xmin
    else:
        minX=min(numpyDicomFlatten)
    if xmax is not None:
        numpyDicomFlatten = numpyDicomFlatten[numpyDicomFlatten <= xmax]
        maxX=xmax
    else:
        maxX=max(numpyDicomFlatten)

    plt.close("all")
    fig, axs = plt.subplots(figsize=(20, 3))

    # axs.set_xlim([xmin, xmax])

    n, bins, patches = plt.hist(numpyDicomFlatten, bins=bins, edgecolor='black', linewidth=0.5)
    # axs.set_ylim(bottom=-axs.get_ylim()[1] / 20)
    # axs.xaxis.set_ticks(np.arange(xmin, xmax, 100))

    axs.set_title(r'Histogram Unidades de Hounsfield')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")

    plt.xticks(np.arange(minX, maxX, ticks))

    fig.tight_layout()
    return plt


def pltHistograma(itkImage, bins=80, xmin=None, xmax=None, ticks=100):
    """
    Shows matplotlib histogram plot
    Muestra un plot de matplotlib con un histograma

    Parameters
    ----------
    itkImage : itkImage
        itk image
        Imagen en formato itk
    bins : int
        Rectangle number in wich the graph is divided
        Número de rectángulos en los que dividir el gráfico
    xmin : int
        Minimum axis x value
        Mínimo valor en el eje x
    xmax : int
        Máximum axis x value
        Máximo valor en el eje x
    ticks : int
        Distance between numbers shown in x axis
        Distancia entre los números que se muestran en el eje x

    Returns
    -------
    None
    """
    numpyDicom = itk.GetArrayViewFromImage(itkImage)
    obtenerHistograma(numpyDicom, bins=bins, xmin=xmin, xmax=xmax, ticks=ticks).show()


def animacionNumpySegmentado(numpyDicom, numpyDicomFiltrado, rutaFichero=None, vmin=-1000, vmax=2000):
    """
    Return animation as matplotlib plot of 2 3D images, one as the base and the other overlaped with alpha.
    Retorna una animación como plot de matplotlib de 2 imágenes 3D, una como base y la otra solapada con transparencia.

    Parameters
    ----------
    numpyDicom : numpy array
        3D image in numpy format. This is the base image on top of wich the othe image will be shown. Usually this
        image will be the original scan.
        Imagen en formato numpy de dimensión 3D. Es la imagen base sobre la que se mostrará la otra.
        Usualmente será el escaneo original sin modificar.
    numpyDicomFiltrado : numpy array
        3D image in numpy format. It will be shown on top of the base image with some alpha. It can be a binary
        mask or a region segmentation.
        Imagen en formato numpy de dimensión 3D. Se mostrará encima de la imagen anterior con algo de transparencia.
        Puede ser tanto una máscara binaria como una segmentación de una región.
    rutaFichero : str
        If it is provided a video with the animation will be stored in the path provided in this parameter.
        En caso de proporcionarse ruta de fichero, se almacenará un video con la animación.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa

    Returns
    -------
    Animation as matplotlib plot
    Animación como plot de matplotlib
    """
    plt.rcParams['animation.embed_limit'] = 2 ** 128

    fig = plt.figure(num="Anim", figsize=(8, 8))

    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi

    ims = []
    for i in range(numpyDicom.shape[0]):
        im1 = plt.imshow(numpyDicom[i], animated=True, cmap=plt.get_cmap("bone"), vmin=vmin, vmax=vmax)
        im2 = plt.imshow(numpyDicomFiltrado[i], animated=True, alpha=0.3, cmap=plt.get_cmap("inferno"))
        text = plt.text(numpyDicom.shape[1] - 40, 25, str(i), horizontalalignment='center',
                        verticalalignment='center', color='w', size=20)

        ims.append([im1, text, im2])


    anim = animation.ArtistAnimation(fig, ims, interval=30,
                                     repeat_delay=0, repeat=True)

    if rutaFichero:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        #         writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #         anim.save('pulmonesSegmentados.mp4', writer=writer)
        writer = Writer(fps=15, codec='libvpx-vp9', bitrate=5800)  # or libvpx-vp8
        anim.save(rutaFichero + ".webm", writer=writer)

    #         anim.save('pulmonesSegmentados.gif', writer='pillow', fps=60)

    return fig


def animacionNumpy(numpyDicom, rutaFichero=None, vmin=-1000, vmax=2000):
    """
    Returns animation of a 3D image slice by slice as matplotlib plot
    Retorna una animación como plot de matplotlib de una imagen 3D.

    Parameters
    ----------
    numpyDicom : numpy array
        3D image in numpy format
        Imagen en formato numpy de dimensión 3D.
    rutaFichero : str
        If a path is provided here, a video containing the animation will be stored there.
        En caso de proporcionarse ruta de fichero, se almacenará un video con la animación.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa

    Returns
    -------
    Animation as matplotlib plot
    Animación como plot de matplotlib
    """
    plt.rcParams['animation.embed_limit'] = 2 ** 128

    fig = plt.figure(num="Anim", figsize=(8, 8))

    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi

    ims = []
    for i in range(numpyDicom.shape[2]):
        im1 = plt.imshow(numpyDicom[:, :, i], animated=True, cmap=plt.get_cmap("bone"), vmin=vmin, vmax=vmax)
        text = plt.text(width - 90, 25, i, horizontalalignment='center',
                        verticalalignment='center', color='w', size=20)
        ims.append([im1, text])

    anim = animation.ArtistAnimation(fig, ims, interval=30,
                                     repeat_delay=0, repeat=True)

    if rutaFichero:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        #         writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #         anim.save('pulmonesSegmentados.mp4', writer=writer)
        writer = Writer(fps=15, codec='libvpx-vp9', bitrate=5800)  # or libvpx-vp8
        anim.save(rutaFichero + ".webm", writer=writer)

    #         anim.save('pulmonesSegmentados.gif', writer='pillow', fps=60)

    return fig


def pltItkImage2D(itkImage, cmap="bone", vmin=None, vmax=None, size=10):
    """
    Plot a 2D image using matplotlib.
    Muestra una imagen 2D en matplotlib.

    Parameters
    ----------
    itkImage : itkImage
        2D itk image
        Imagen 2D en formato itkImage.
    cmap : str
        Name of the colormap used to represent Hounsfield Units of image.
        Mapa de color de matplotlib utilizado para representar la escala de instensidades hounsfield de la imagen.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    size : int
        Size of the image to show
        Tamaño de la imagen a mostrar.

    Returns
    -------
    None
    """
    numpyView = itk.GetArrayViewFromImage(itkImage)

    fig, ax = plt.subplots(figsize=(size, size))

    plt.imshow(numpyView, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.show()


def pltItkImage(image, slice=0, cmap="bone", vmin=None, vmax=None, size=5, show=True):
    """
    Plot a 2D image or an axial slice of a 3D image using matplotlib.
    Muestra una imagen 2D o una capa axial de una imagen 3D en matplotlib.

    Parameters
    ----------
    image : itkImage | numpy array
        2D or 3D image in numpy or in itk format.
        Imagen 2D o 3D en formato itkImage o numpy array.
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    cmap : str
        Name of the colormap used to represent Hounsfield Units of image.
        Mapa de color de matplotlib utilizado para representar la escala de instensidades hounsfield de la imagen.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.
    show : boolean
        Flag that controlls if matplotlib show method will be called to show the plot. This mechanism is designed for jupyter
        notebooks and jupyterlab, where if called show, the image is shown 2 times.
        Flag que controla si se ejecuta el método show de matplotlib para mostrar el gráfico. Es útil en notebooks,
        ya que a veces si se ejecuta show se muestra la imagen repetida.

    Returns
    -------
    None
    """
    if type(image) == ImageTypeSS3 or type(image) == ImageTypeUS3:
        image = itk.GetArrayViewFromImage(image)

    if vmin is None or vmax is None:
        if np.amin(image) == 0 and np.amax(image) == 1:
            vmin = 0
            vmax = 1
        else:
            vmin = -1000
            vmax = 2000

    fig, ax = plt.subplots(figsize=(size, size))

    if len(image.shape) == 3:
        image = image[slice]
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if show:
        plt.show()


def pltItkImageRectangle(itkImage, slice=0 , boundingBox=None, rectangle=None, cmap="bone", vmin=None, vmax=None, size=10, lineWidth=3, pointWidth=20):
    """
    Plot a 2D image or an axial slice of a 3D image using matplotlib with an overlap rectangle and a dot in the center.
    Muestra una imagen 2D o una capa axial de una imagen 3D en matplotlib con un rectángulo encima y un punto en el
    centro del rectángulo.

    Parameters
    ----------
    image : itkImage | numpy array
        2D or 3D image in numpy or in itk format.
        Imagen 2D o 3D en formato itkImage o numpy array.
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    boundingBox : itkRegion
        Region that defines the dimension and position of the rectangle to be drawn in itkRegion format. If not defined is 
        necesary to define rectangle parameter.
        Region en la que dibujar el rectángulo en formato itkRegion. Si no se define es necesario definir rectangle.
    rectangle : list(x, y, w, h)
        Rectangle represented as a list.
        Rectángulo representado como lista, con la posición x e y iniciales así como el ancho y alto del rectángulo.
        todo fusionar este campo con el anterior de boundingBox. 
    cmap : str
        Name of the colormap used to represent Hounsfield Units of image.
        Mapa de color de matplotlib utilizado para representar la escala de instensidades hounsfield de la imagen.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.

    Returns
    -------
    None
    """
    

    if boundingBox is not None:
        x = boundingBox.GetIndex()[0]
        y = boundingBox.GetIndex()[1]
        width = boundingBox.GetSize()[0]
        height = boundingBox.GetSize()[1]
    elif rectangle is not None:
        x = rectangle[0]
        y = rectangle[1]
        width = rectangle[2]
        height = rectangle[3]

    # Corrigiendo posición para extracciones
    imageRegionIndex = itkImage.GetLargestPossibleRegion().GetIndex()
    # print(imageRegionIndex)
    x -= imageRegionIndex[0]+1
    y -= imageRegionIndex[1]+1

    if type(itkImage) == np.ndarray:
        sliceToShow = itkImage[slice]
    else:
        sliceToShow = itk.GetArrayViewFromImage(itkImage)[slice]

    # print(sliceToShow)

    if vmin is None or vmax is None:
        if np.amin(sliceToShow) == 0 and np.amax(sliceToShow) == 1:
            vmin = 0
            vmax = 1
        else:
            vmin = -1000
            vmax = 2000

    fig, ax = plt.subplots(figsize=(size, size))
    # print(sliceToShow.shape)
    plt.imshow(sliceToShow, cmap=cmap, vmin=vmin, vmax=vmax)

    if boundingBox is not None or rectangle is not None:
        rect = patches.Rectangle(xy=(x, y), width=width, height=height, linewidth=lineWidth,  edgecolor='r', facecolor='none')

        ax.add_patch(rect)
        plt.scatter(x + width/2, y + height/2, s=pointWidth, c="red")
    plt.show()


def pltItkImageDrawPoint(itkImage, slice=0, point2D=None, cmap="bone", vmin=None, vmax=None, size=10):
    """
    Shows an axial slice of a 3D image with a dot overlaped.
    Muestra una capa axial de una imagen 3D en matplotlib con un punto encima.

    Parameters
    ----------
    itkImage : itkImage
        3D itk image
        Imagen 3D en formato itkImage
    slice : int
        Axial slice from the image to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    point2D : list(x,y)
        Coords of the dot to be drawn on top of the slice.
        Coordenadas del punto a dibujar.
    cmap : str
        Name of the colormap used to represent Hounsfield Units of image.
        Mapa de color de matplotlib utilizado para representar la escala de instensidades hounsfield de la imagen.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.

    Returns
    -------
    None
    """
    x=200
    y=200
    if point2D is not None:
        x=point2D[0]
        y=point2D[1]

    numpyView = itk.GetArrayViewFromImage(itkImage)

    fig, ax = plt.subplots(figsize=(size, size))

    plt.imshow(numpyView[slice], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.scatter(x, y, s=20, c="red")
    plt.show()


def pltItkImageInteractivo(itkImage, slice=0, vmin=None, vmax=None, size=8):
    """
    Shows a 2D image or an axial slice of a 3D image in a jupyterlab notebook with a slider that lets interactively
    change the slice shown.
    Muestra una imagen 2D o una capa axial de una imagen 3D en matplotlib de forma interactiva en un notebook de
    jupyter lab. Para ello genera un slider y la imagen, controlando el slider la capa axial a mostrar.

    Parameters
    ----------
    itkImage : itkImage | numpy array
        2D or 3D image in numpy or in itk format.
        Imagen 2D o 3D en formato itkImage o numpy array.
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.

    Returns
    -------
    None
    """
    if type(itkImage) != np.ndarray:
        image = itk.GetArrayViewFromImage(itkImage)
    else:
        image = itkImage

    if vmin is None or vmax is None:
        if np.amin(image) == 0 and np.amax(image) == 1:
            vmin = 0
            vmax = 1
        else:
            vmin = -1000
            vmax = 2000

    def pltsin(slice=slice):
        plt.close("all")
        fig, ax = plt.subplots(figsize=(size, size))
        im = plt.imshow(image[slice], cmap=plt.get_cmap("bone"), vmin=vmin, vmax=vmax, interpolation='nearest')
        #plt.tight_layout()
        plt.show()

    interact(pltsin, slice=(0,image.shape[0]-1,1))

    # Para que se almacene la capa inicial en jupyter lab
    plt.subplots(figsize=(4, 4))
    plt.imshow(image[slice], cmap=plt.get_cmap("bone"), vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.show()


def pltItkImageInteractivoRectangle(itkImage, slice=0, vmin=None, vmax=None, size=5, boundingBoxes=None, rectangle=None, itkImageBoundingBoxes=None, lineWidth=3, pointWidth=20):
    """
    Shows an axial slice of a 3D image with rectangles overlaped in a jupyterlab notebook with a slider that lets interactively
    change the slice shown. 
    Muestra una capa axial 2D de una imagen 3D en matplotlib con rectángulos encima de forma interactiva
    en un notebook de jupyter lab. Para ello genera un slider y la imagen, controlando el slider la capa axial a mostrar.

    Parameters
    ----------
    itkImage : itkImage | numpy array
        2D or 3D image in numpy or in itk format.
        Imagen 3D
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.
    boundingBoxes : list(itkRegion1, ..., itkRegionN) | itkRegion
        Rectangles to be drawn on top of the image. If empty itkImageBoundingBoxes has to be provided.
        Información sobre los rectángulos a dibujar encima de la imagen. Esta información se puede proporcionar en lugar
        de con este parámetro con el siguiente de itkImageBoundingBoxes, solo se utilizará uno para dibujar los rectángulos.
    itkImageBoundingBoxes : itkImage
        3D itk image containing a binary mask and with the same dimensions as itkImage parameter. Bounding boxes will
        be extracted slice by slice.
        Imagen en formato itkImage en 3D de las mismas dimensiones que itkImage del primer parámetro. Debe ser una
        máscara binaria. De esta, se extraen los boundingBox por capas para mostrar el rectangulo de la capa actual que
        contiene la máscara binaria.

    Returns
    -------
    None
    """
    numpyView = itk.GetArrayViewFromImage(itkImage)



    def pltsin(slice=slice):
        boundingBox=None
        plt.close("all")
        if itkImageBoundingBoxes is not None:
            itk3dSlice = extractSlice(itkImageBoundingBoxes, sliceNum=slice)
            boundingBox = getBoundingBoxes(itk3dSlice)[0]
        elif boundingBoxes is not None:
            if type(boundingBoxes) == list:
                boundingBox = boundingBoxes[slice]
            else:
                if boundingBoxes.GetIndex()[2] <= slice <= boundingBoxes.GetIndex()[2]+boundingBoxes.GetSize()[2]:
                    boundingBox = boundingBoxes
                else:
                    boundingBox = None
        elif rectangle is None:
            itk3dSlice = extractSlice(itkImage, sliceNum=slice)
            boundingBox = getBoundingBoxes(itk3dSlice)[0]
        else:
            boundingBox=rectangle
            
            
        pltItkImageRectangle(itkImage, slice=slice, size=size, boundingBox=boundingBox, rectangle=rectangle, vmin=vmin, vmax=vmax, lineWidth=lineWidth, pointWidth=pointWidth)

    interact(pltsin, slice=(0,numpyView.shape[0]-1,1))


def pltItkImageCompareInteractivo(itkImage1, itkImage2, slice=0, vmin1=None, vmax1=None, cmap1="bone", vmin2=None, vmax2=None, cmap2="bone", size=15):
    """
    Shows a 2D axial slice of a 3D image at the left and another at the right a jupyterlab notebook with a slider that lets interactively
    change the slice shown.
    Muestra una capa axial 2D de una imagen 3D en matplotlib a la izquierda y otra capa axial 2D de otra imagen 3D a la derecha,
    de forma interactiva, en un notebook de jupyter lab. Para ello genera un slider y las imagenes, controlando el slider
    la capa axial de ambas imágenes a mostrar.

    Parameters
    ----------
    itkImage1 : itkImage
        3D image show at left.
        Imagen 3D en formato itkImage que se mostrará a la izquierda.
    itkImage2 : itkImage
        3D image show at right.
        Imagen 3D en formato itkImage que se mostrará a la derecha.
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.

    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa

    vmin1 : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices. Used with itkImage1.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa. Utilizado para itkImage1.
    vmax1 : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices. Used with itkImage1.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa. Utilizado para itkImage1.
    cmap1 : string
        Colormap utilizado para mostrar itkImage1.
    vmin2 : int
        Same as vmin1 for itkImage2
        Igual que vmin1 para itkImage2.
    vmax2 : int
        Same as vmin1 for itkImage2
        Igual que vmax1 para itkImage2.
    cmap2 : string
        Same as vmin1 for itkImage2
        Igual que cmap1 para itkImage2.
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.

    Returns
    -------
    None
    """
    numpyView1 = itk.GetArrayViewFromImage(itkImage1)
    numpyView2 = itk.GetArrayViewFromImage(itkImage2)

    if vmin1 is None or vmax1 is None:
        if np.amin(numpyView1) == 0 and np.amax(numpyView1) == 1:
            vmin1 = vmin2 = 0
            vmax1 = vmax2 = 1
        else:
            vmin1 = vmin2 = -1000
            vmax1 = vmax2 = 2000

    def pltsin(slice=slice):
        plt.close("all")
        fig, ax = plt.subplots(1,2,figsize=(size, size))
        ax[0].imshow(numpyView1[slice], cmap=cmap1, vmin=vmin1, vmax=vmax1, interpolation='nearest')
        ax[1].imshow(numpyView2[slice], cmap=cmap2, vmin=vmin2, vmax=vmax2, interpolation='nearest')
        #plt.tight_layout()
        plt.show()

    interact(pltsin, slice=(0,numpyView1.shape[0]-1,1))

    # Para que se almacene la capa inicial en jupyterlab
    fig, ax = plt.subplots(1,2,figsize=(10, 10))
    ax[0].imshow(numpyView1[slice], cmap=cmap1, vmin=vmin1, vmax=vmax1, interpolation='nearest')
    ax[1].imshow(numpyView2[slice], cmap=cmap2, vmin=vmin2, vmax=vmax2, interpolation='nearest')
    plt.show()

def pltItkImageOverlap(itkImage, itkImageOverlap, itkImageOverlap2=None, slice=0, vmin=None, vmax=None, cmap1="bone", vmin2=None, vmax2=None, cmap2="Wistia", cmap3="jet", alpha=0.5, size=5, show=True):
    """
    Plot 2D axial slice of a 3D image with 1 or 2 axial slices from other 3D images overlaped.
    Muestra una capa axial 2D de una imagen 3D en matplotlib con otra capa o capas axiales 2D de otras imagenes 3D
    solapadas encima, en un notebook de jupyter lab.

    Parameters
    ----------
    itkImage : itkImage | numpyArray
        3D base image.
        Imagen 3D sobre la que se solapará otra imagen u imágenes.
    itkImageOverlap : itkImage | numpyArray
        3D image that overlaps the itkImage
        Imagen 3D que se solapará sobre itkImage con cierto grado de transparencia.
    itkImageOverlap2 : itkImage | numpyArray
        3D image that overlaps itkImage and itkImageOverlap
        Imagen 3D que se solapará sobre itkImage y sobre itkImageOvelap con cierto grado de transparencia.
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    cmap1 : string
        Colormap used for itkImage
        Colormap utilizado para mostrar itkImage.
    cmap2 : string
        Colormap used for itkImageOverlap
        Igual que cmap1 para itkImageOverlap.
    cmap3 : string
        Colormap used for itkImageOverlap2
        Igual que cmap1 para itkImageOverlap2.
    alpha : float
        Transparency value used in overlaped images. It is a value in the range [0-1] being 0 transparent and 1 opaque.
        Valor de transparencia de las imágenes solapadas indicado en intervalo [0-1] siendo 0 transparente y 1 opaca.
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.
    show : boolean
        Flag that controlls if matplotlib show method will be called to show the plot. This mechanism is designed for jupyter
        notebooks and jupyterlab, where if called show, the image is shown 2 times.
        Flag que controla si se ejecuta el método show de matplotlib para mostrar el gráfico. Es útil en notebooks,
        ya que a veces si se ejecuta show se muestra la imagen repetida.
        todo mejorar para que devuelva la figura y se pueda utilizar posteriormente para otras utilizades como almacenarla.

    Returns
    -------
    None
    """
    if type(itkImage) == np.ndarray:
        numpyView = itkImage
    else: 
        numpyView = itk.GetArrayViewFromImage(itkImage)[slice]

    if vmin is None or vmax is None:
        if np.amin(numpyView) == 0 and np.amax(numpyView) == 1:
            vmin = 0
            vmax = 1
        else:
            vmin = -1000
            vmax = 2000

    if type(itkImageOverlap) == np.ndarray:
        numpyViewOverlap = itkImageOverlap
    else:
        numpyViewOverlap = itk.GetArrayFromImage(itkImageOverlap)[slice]

    if itkImageOverlap2:
        if type(itkImageOverlap2) == np.ndarray:
            numpyViewOverlap2 = itkImageOverlap2
        else:
            numpyViewOverlap2 = itk.GetArrayFromImage(itkImageOverlap2)[slice]

    vmin2 = np.nanmin(numpyViewOverlap)
    vmax2 = np.nanmax(numpyViewOverlap)

    numpyViewOverlap = np.where(numpyViewOverlap==0, np.nan, numpyViewOverlap)

    fig, ax = plt.subplots(figsize=(size, size))

    plt.imshow(numpyView, cmap=cmap1, vmin=vmin, vmax=vmax)
    plt.imshow(numpyViewOverlap, alpha=alpha, cmap=cmap2, vmin=vmin2, vmax=vmax2)
    if itkImageOverlap2:
        plt.imshow(numpyViewOverlap2, alpha=alpha, cmap=cmap3)

    if show:
        plt.show()

def pltItkImageOverlapInteractivo(itkImage, itkImageOverlap, itkImageOverlap2=None, slice=0, vmin=None, vmax=None, vmin2=None, vmax2=None, size=5, cmap1="bone", cmap2="Wistia", cmap3="jet", alpha=0.5):
    """
    Plot 2D axial slice of a 3D image with 1 or 2 axial slices from other 3D images overlaped with an interactive slider in jupyterlab notebook.
    Muestra una capa axial 2D de una imagen 3D en matplotlib con otra capa o capas axiales 2D de otras imagenes 3D solapadas encima,
    de forma interactiva, en un notebook de jupyter lab. Para ello genera un slider y la imagen, controlando el slider
    la capa axial de ambas imágenes a mostrar.

    Parameters
    ----------
    itkImage : itkImage | numpyArray
        3D base image.
        Imagen 3D sobre la que se solapará otra imagen u imágenes.
    itkImageOverlap : itkImage | numpyArray
        3D image that overlaps the itkImage
        Imagen 3D que se solapará sobre itkImage con cierto grado de transparencia.
    itkImageOverlap2 : itkImage | numpyArray
        3D image that overlaps itkImage and itkImageOverlap
        Imagen 3D que se solapará sobre itkImage y sobre itkImageOvelap con cierto grado de transparencia.
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    cmap1 : string
        Colormap used for itkImage
        Colormap utilizado para mostrar itkImage.
    cmap2 : string
        Colormap used for itkImageOverlap
        Igual que cmap1 para itkImageOverlap.
    cmap3 : string
        Colormap used for itkImageOverlap2
        Igual que cmap1 para itkImageOverlap2.
    alpha : float
        Transparency value used in overlaped images. It is a value in the range [0-1] being 0 transparent and 1 opaque.
        Valor de transparencia de las imágenes solapadas indicado en intervalo [0-1] siendo 0 transparente y 1 opaca.
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.

    Returns
    -------
    None
    """

    if type(itkImageOverlap) == np.ndarray:
        numpyViewOverlap = itkImageOverlap
    else:
        numpyViewOverlap = itk.GetArrayFromImage(itkImageOverlap)[slice]

    vmin2 = np.nanmin(numpyViewOverlap)
    vmax2 = np.nanmax(numpyViewOverlap)

    def pltsin(slice=slice):
        plt.close("all")
        pltItkImageOverlap(itkImage, itkImageOverlap,itkImageOverlap2=itkImageOverlap2, slice=slice, size=size, vmin=vmin, vmax=vmax, alpha=alpha, cmap1=cmap1, vmin2=vmin2, vmax2=vmax2, cmap2=cmap2, cmap3=cmap3)

    interact(pltsin, slice=(0,itkImage.GetLargestPossibleRegion().GetSize()[2]-1,1))
    pltItkImageOverlap(itkImage, numpyViewOverlap, itkImageOverlap2=itkImageOverlap2, slice=slice, size=4, vmin=vmin,
                       vmax=vmax, alpha=alpha, cmap1=cmap1, vmin2=vmin2, vmax2=vmax2, cmap2=cmap2, cmap3=cmap3)


def viewRecord(itkImage, filename="video"):
    """
    Shows a viewer from itk-widgets to visualize itk image with volume rendering, and also a button to record the content
    of the viewer.
    Muestra un visualizador de itk-widgets para ver una imagen itk en volume rendering, y un botón para grabar el contenido
    de este visualizador. Una vez grabado, permite ver el video y almacenarlo.

    Parameters
    ----------
    itkImage : itkImage
        3D image shown in the viewer.
        Imagen 3D en formato itkImage que se mostrará con volume rendering.
    filename : string
        Path of the file to store.
        Ruta del fichero a almacenar.

    Returns
    -------
    None
    """
    filename += ".webm"
    viewer = view(itkImage, cmap=itkwidgets.cm.BuRd, annotations=False)
    display(viewer)
    recorder = webrtc.VideoRecorder(stream=viewer, filename=filename, autosave=True)
    display(recorder)


def pltItkImageOverlapRectangle(itkImage, itkImageOverlap, itkImageOverlap2=None, slice=0, boundingBox=None, rectangle=None, vmin=None, vmax=None, vmin2=None, vmax2=None, cmap1="bone", cmap2="Wistia", cmap3="jet", alpha=0.5, size=5, show=True):
    """
    Plot 2D axial slice of a 3D image with 1 or 2 axial slices from other 3D images overlaped, with rectangles overlaped and with an interactive slider in jupyterlab notebook.

    Muestra una capa axial 2D de una imagen 3D en matplotlib con otra capa axial 2D de otra imagen 3D solapada encima,
    de forma interactiva y un rectángulo solapado encima, en un notebook de jupyter lab. Para ello genera un slider y la imagen, controlando el slider
    la capa axial de ambas imágenes a mostrar.

    Parameters
    ----------
    itkImage : itkImage | numpyArray
        3D base image.
        Imagen 3D sobre la que se solapará otra imagen u imágenes.
    itkImageOverlap : itkImage | numpyArray
        3D image that overlaps the itkImage
        Imagen 3D que se solapará sobre itkImage con cierto grado de transparencia.
    itkImageOverlap2 : itkImage | numpyArray
        3D image that overlaps itkImage and itkImageOverlap
        Imagen 3D que se solapará sobre itkImage y sobre itkImageOvelap con cierto grado de transparencia.
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    boundingBox : itkRegion
        Region that defines the dimension and position of the rectangle to be drawn in itkRegion format. If not defined is 
        necesary to define rectangle parameter.
        Region en la que dibujar el rectángulo en formato itkRegion. Si no se define es necesario definir rectangle.
    rectangle : list(x, y, w, h)
        Rectangle represented as a list.
        Rectángulo representado como lista, con la posición x e y iniciales así como el ancho y alto del rectángulo.
        todo fusionar este campo con el anterior de boundingBox. 
    vmin : int
        Minimum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor mínimo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    vmax : int
        Maximum value of Hounsfield Units in the visualization asociated with colormap. It is used so the video shows the same color values
        for the same HUs along all slices.
        Valor máximo de intensidad hounsfield de cara a visualización. Se utiliza para que el video entero presente
        los mismos colores y no varíe capa a capa
    cmap1 : string
        Colormap used for itkImage
        Colormap utilizado para mostrar itkImage.
    cmap2 : string
        Colormap used for itkImageOverlap
        Igual que cmap1 para itkImageOverlap.
    cmap3 : string
        Colormap used for itkImageOverlap2
        Igual que cmap1 para itkImageOverlap2.
    alpha : float
        Transparency value used in overlaped images. It is a value in the range [0-1] being 0 transparent and 1 opaque.
        Valor de transparencia de las imágenes solapadas indicado en intervalo [0-1] siendo 0 transparente y 1 opaca.
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.
    show : boolean
        Flag that controlls if matplotlib show method will be called to show the plot. This mechanism is designed for jupyter
        notebooks and jupyterlab, where if called show, the image is shown 2 times.
        Flag que controla si se ejecuta el método show de matplotlib para mostrar el gráfico. Es útil en notebooks,
        ya que a veces si se ejecuta show se muestra la imagen repetida.
        todo mejorar para que devuelva la figura y se pueda utilizar posteriormente para otras utilizades como almacenarla.

    Returns
    -------
    None
    """
    if boundingBox is not None:
        x = boundingBox.GetIndex()[0]
        y = boundingBox.GetIndex()[1]
        width = boundingBox.GetSize()[0]
        height = boundingBox.GetSize()[1]
    elif rectangle is not None:
        x = rectangle[0]
        y = rectangle[1]
        width = rectangle[2]
        height = rectangle[3]

    if type(itkImage) == np.ndarray:
        numpyView = itkImage
    else:
        numpyView = itk.GetArrayViewFromImage(itkImage)[slice]

        # Corrigiendo posición para extracciones
        imageRegionIndex = itkImage.GetLargestPossibleRegion().GetIndex()
        # print(imageRegionIndex)
        x -= imageRegionIndex[0]+1
        y -= imageRegionIndex[1]+1

    if vmin is None or vmax is None:
        if np.amin(numpyView) == 0 and np.amax(numpyView) == 1:
            vmin = 0
            vmax = 1
        else:
            vmin = -1000
            vmax = 2000

    if type(itkImageOverlap) == np.ndarray:
        numpyViewOverlap = itkImageOverlap
    else:
        numpyViewOverlap = itk.GetArrayFromImage(itkImageOverlap)[slice]

    vmin2 = np.nanmin(numpyViewOverlap)
    vmax2 = np.nanmax(numpyViewOverlap)

    if itkImageOverlap2:
        if type(itkImageOverlap2) == np.ndarray:
            numpyViewOverlap2 = itkImageOverlap2
        else:
            numpyViewOverlap2 = itk.GetArrayFromImage(itkImageOverlap2)[slice]

    numpyViewOverlap = np.where(numpyViewOverlap==0, np.nan, numpyViewOverlap)

    fig, ax = plt.subplots(figsize=(size, size))

    plt.imshow(numpyView, cmap=cmap1, vmin=vmin, vmax=vmax)
    plt.imshow(numpyViewOverlap, alpha=alpha, cmap=cmap2, vmin=vmin2, vmax=vmax2)
    if itkImageOverlap2:
        plt.imshow(numpyViewOverlap2, alpha=alpha, cmap=cmap3)

    if boundingBox is not None or rectangle is not None:
        rect = patches.Rectangle(xy=(x, y), width=width, height=height, linewidth=3,  edgecolor='r', facecolor='none')

        ax.add_patch(rect)
        plt.scatter(x + width/2, y + height/2, s=20, c="red")

    if show:
        plt.show()


def pltItkImageOverlapMultiple(listElements, slice=0, size=5, show=True):
    """
    Plot 2D axial slice of a 3D image with several axial slices from other 3D images overlaped with an interactive slider in jupyterlab notebook.

    Muestra una capa axial 2D de una imagen 3D en matplotlib con otra capa o capas axiales 2D de otras imagenes 3D
    solapadas encima, en un notebook de jupyter lab. Utiliza para ello una lista de elementos que contiene toda la
    información a mostrar

    Parameters
    ----------
    listElements : list(list1(itkImage1, vmin1, vmax1, cmap1, alpha1), ..., listN(itkImageN, vminN, vmaxN, cmapN, alphaN))
        List of elements used to show the overlaped images. For reference, read pltkItkImageOverlap description
        Lista de elementos utilizada para mostrar todas las imágenes solapadas. Para referencia de que es cada elemento
        leer la documentación de la funcion pltkItkImageOverlap
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.
    show : boolean
        Flag that controlls if matplotlib show method will be called to show the plot. This mechanism is designed for jupyter
        notebooks and jupyterlab, where if called show, the image is shown 2 times.
        Flag que controla si se ejecuta el método show de matplotlib para mostrar el gráfico. Es útil en notebooks,
        ya que a veces si se ejecuta show se muestra la imagen repetida.
        todo mejorar para que devuelva la figura y se pueda utilizar posteriormente para otras utilizades como almacenarla.

    Returns
    -------
    None
    """

    posItkImage = 0
    posVmin = 1
    posVmax = 2
    posCmap = 3
    posAlpha = 4

    plt.subplots(figsize=(size, size))

    for element in listElements:
        itkImage = element[posItkImage]
        vmin = element[posVmin]
        vmax = element[posVmax]
        cmap = element[posCmap]
        alpha = element[posAlpha]

        if type(itkImage) == np.ndarray:
            numpyView = itkImage
        else:
            numpyView = itk.GetArrayViewFromImage(itkImage)[slice]

        if vmin is None or vmin is None:
            if np.amin(numpyView) == 0 and np.amax(numpyView) == 1:
                vmin = 0
                vmax = 1
            else:
                vmin = -1000
                vmax = 2000

        numpyView = np.where(numpyView == -3000, np.nan, numpyView)

        plt.imshow(numpyView, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

    if show:
        plt.show()


def pltItkImageOverlapMultipleInteractivo(listElements, slice=0, size=5, show=True):
    """
    pltItkImageOverlapMultiple with interactive slider in jupyterlab.
    Muestra una capa axial 2D de una imagen 3D en matplotlib con otra capa o capas axiales 2D de otras imagenes 3D
    solapadas encima, en un notebook de jupyter lab de forma interactiva. Utiliza para ello una lista de elementos
    que contiene toda la información a mostrar

    Parameters
    ----------
    listElements : list(list1(itkImage1, vmin1, vmax1, cmap1, alpha1), ..., listN(itkImageN, vminN, vmaxN, cmapN, alphaN))
        List of elements used to show the overlaped images. For reference, read pltkItkImageOverlap description
        Lista de elementos utilizada para mostrar todas las imágenes solapadas. Para referencia de que es cada elemento
        leer la documentación de la funcion pltkItkImageOverlap
    slice : int
        Used in 3D images as the axial slice to be shown.
        Se utiliza en imágenes 3D, es la capa axial a visualizar.
    size : int
        Size of the output image that will be shown.
        Tamaño de la imagen a mostrar.
    show : boolean
        Flag that controlls if matplotlib show method will be called to show the plot. This mechanism is designed for jupyter
        notebooks and jupyterlab, where if called show, the image is shown 2 times.
        Flag que controla si se ejecuta el método show de matplotlib para mostrar el gráfico. Es útil en notebooks,
        ya que a veces si se ejecuta show se muestra la imagen repetida.
        todo mejorar para que devuelva la figura y se pueda utilizar posteriormente para otras utilizades como almacenarla.

    Returns
    -------
    None
    """

    # if type(itkImageOverlap) == np.ndarray:
    #     numpyViewOverlap = itkImageOverlap
    # else:
    #     numpyViewOverlap = itk.GetArrayFromImage(itkImageOverlap)[slice]
    #
    # vmin2 = np.nanmin(numpyViewOverlap)
    # vmax2 = np.nanmax(numpyViewOverlap)

    def pltsin(slice=slice):
        plt.close("all")
        pltItkImageOverlapMultiple(listElements=listElements, slice=slice, size=size)

    interact(pltsin, slice=(0, listElements[0][0].GetLargestPossibleRegion().GetSize()[2]-1, 1))
    pltItkImageOverlapMultiple(listElements=listElements, slice=slice, size=4)  # Para que se guarde en los notebooks


def pltItkImageOverlapMultipleAnim(listElements, rutaFichero=None):
    """
    Generates animation for pltItkImageOverlapMultiple
    Genera una animación con pltItkImageOverlapMultiple

    Parameters
    ----------
    listElements : list(list1(itkImage1, vmin1, vmax1, cmap1, alpha1), ..., listN(itkImageN, vminN, vmaxN, cmapN, alphaN))
        List of elements used to show the overlaped images. For reference, read pltkItkImageOverlap description
        Lista de elementos utilizada para mostrar todas las imágenes solapadas. Para referencia de que es cada elemento
        leer la documentación de la funcion pltkItkImageOverlap

    rutaFichero : str
        If a path is provided here, a video containing the animation will be stored there.
        En caso de proporcionarse ruta de fichero, se almacenará un video con la animación.
        
    Returns
    -------
    None
    """
    plt.rcParams['animation.embed_limit'] = 2 ** 128

    posItkImage = 0
    posVmin = 1
    posVmax = 2
    posCmap = 3
    posAlpha = 4

    listElements = [list(i) for i in listElements]   # De tupla a lista

    for element in listElements:
        itkImage = element[posItkImage]
        vmin = element[posVmin]
        vmax = element[posVmax]

        if type(itkImage) == np.ndarray:
            element[posItkImage] = itkImage
        else:
            element[posItkImage] = itk.GetArrayViewFromImage(itkImage)

        if vmin is None or vmin is None:
            if np.amin(element[posItkImage]) == 0 and np.amax(element[posItkImage]) == 1:
                element[posVmin] = 0
                vmax = 1
            else:
                element[posVmin] = -1000
                element[posVmax] = 2000

        element[posItkImage] = np.where(element[posItkImage] == -3000, np.nan, element[posItkImage])

    fig = plt.figure(num="Anim", figsize=(8, 8))

    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi

    ims = []
    for i in range(listElements[0][0].shape[0]):
        imgsToAppend = []

        for element in listElements:
            numpyView = element[posItkImage]
            vmin = element[posVmin]
            vmax = element[posVmax]
            cmap = element[posCmap]
            alpha = element[posAlpha]

            im = plt.imshow(numpyView[i], cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

            imgsToAppend.append(im)

        text = plt.text(listElements[0][0].shape[1] - 40, 25, str(i), horizontalalignment='center',
                        verticalalignment='center', color='w', size=20)

        imgsToAppend.append(text)

        ims.append(imgsToAppend)

    anim = animation.ArtistAnimation(fig, ims, interval=30,
                                     repeat_delay=0, repeat=True)

    if rutaFichero:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        #         writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #         anim.save('pulmonesSegmentados.mp4', writer=writer)
        writer = Writer(fps=15, codec='libvpx-vp9', bitrate=5800)  # or libvpx-vp8
        anim.save(rutaFichero + ".webm", writer=writer)

    #         anim.save('pulmonesSegmentados.gif', writer='pillow', fps=60)

    return fig