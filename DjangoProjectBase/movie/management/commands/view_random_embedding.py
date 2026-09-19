import numpy as np
from django.core.management.base import BaseCommand

from movie.models import Movie


class Command(BaseCommand):
    help = "Display the stored embedding of a randomly selected movie"

    def handle(self, *args, **kwargs):
        movie = Movie.objects.order_by("?").first()

        if movie is None:
            self.stderr.write("No movies found in the database.")
            return

        embedding_vector = np.frombuffer(movie.emb, dtype=np.float32)

        self.stdout.write(self.style.SUCCESS(f"Movie: {movie.title}"))
        self.stdout.write(f"Embedding dimensions: {embedding_vector.size}")
        self.stdout.write(f"First 10 values: {embedding_vector[:10]}")
