import os
from pathlib import Path

import numpy as np
from django.shortcuts import render
from dotenv import load_dotenv
from openai import OpenAI

from .models import Movie

import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64


def cosine_similarity(first, second):
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        return None
    return float(np.dot(first, second) / (first_norm * second_norm))


def recommendation(request):
    prompt = request.POST.get('prompt', '').strip()
    best_movie = None
    best_similarity = None
    error = None

    if request.method == 'POST' and prompt:
        load_dotenv(Path(__file__).resolve().parents[2] / 'openAI.env')
        api_key = os.environ.get('openai_apikey')

        if not api_key:
            error = 'No se encontró la API key de OpenAI en openAI.env.'
        else:
            try:
                client = OpenAI(api_key=api_key)
                response = client.embeddings.create(
                    input=[prompt],
                    model='text-embedding-3-small',
                )
                prompt_embedding = np.asarray(
                    response.data[0].embedding,
                    dtype=np.float32,
                )

                for movie in Movie.objects.exclude(emb__isnull=True):
                    movie_embedding = np.frombuffer(movie.emb, dtype=np.float32)
                    if movie_embedding.shape != prompt_embedding.shape:
                        continue
                    similarity = cosine_similarity(prompt_embedding, movie_embedding)
                    if similarity is not None and (
                        best_similarity is None or similarity > best_similarity
                    ):
                        best_movie = movie
                        best_similarity = similarity

                if best_movie is None:
                    error = 'No hay películas con embeddings compatibles para comparar.'
            except Exception:
                error = 'No fue posible generar la recomendación. Revisa la API key y vuelve a intentarlo.'
    elif request.method == 'POST':
        error = 'Escribe una descripción para buscar una película.'

    return render(request, 'recommendation.html', {
        'prompt': prompt,
        'best_movie': best_movie,
        'best_similarity': best_similarity,
        'error': error,
    })

def home(request):
    #return HttpResponse('<h1>Welcome to Home Page</h1>')
    #return render(request, 'home.html')
    #return render(request, 'home.html', {'name':'Paola Vallejo'})
    searchTerm = request.GET.get('searchMovie') # GET se usa para solicitar recursos de un servidor
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm':searchTerm, 'movies':movies})


def about(request):
    #return HttpResponse('<h1>Welcome to About Page</h1>')
    return render(request, 'about.html')

def signup(request):
    email = request.GET.get('email') 
    return render(request, 'signup.html', {'email':email})


def statistics_view0(request):
    matplotlib.use('Agg')
    # Obtener todas las películas
    all_movies = Movie.objects.all()

    # Crear un diccionario para almacenar la cantidad de películas por año
    movie_counts_by_year = {}

    # Filtrar las películas por año y contar la cantidad de películas por año
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    # Ancho de las barras
    bar_width = 0.5
    # Posiciones de las barras
    bar_positions = range(len(movie_counts_by_year))

    # Crear la gráfica de barras
    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')

    # Personalizar la gráfica
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)

    # Ajustar el espaciado entre las barras
    plt.subplots_adjust(bottom=0.3)

    # Guardar la gráfica en un objeto BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convertir la gráfica a base64
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    # Renderizar la plantilla statistics.html con la gráfica
    return render(request, 'statistics.html', {'graphic': graphic})

def statistics_view(request):
    matplotlib.use('Agg')
    # Gráfica de películas por año
    all_movies = Movie.objects.all()
    movie_counts_by_year = {}
    for movie in all_movies:
        print(movie.genre)
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    # Gráfica de películas por género
    movie_counts_by_genre = {}
    for movie in all_movies:
        # Obtener el primer género
        genres = movie.genre.split(',')[0].strip() if movie.genre else "None"
        if genres in movie_counts_by_genre:
            movie_counts_by_genre[genres] += 1
        else:
            movie_counts_by_genre[genres] = 1

    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {'year_graphic': year_graphic, 'genre_graphic': genre_graphic})


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic