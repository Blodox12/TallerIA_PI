import os
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from movie.models import Movie


class Command(BaseCommand):
    help = "Update movie image fields from files in media/movie/images/"

    def handle(self, *args, **kwargs):
        images_folder = Path(settings.MEDIA_ROOT) / "movie" / "images"

        if not images_folder.is_dir():
            self.stderr.write(f"Images folder not found: {images_folder}")
            return

        image_files = {
            image_file.name: image_file
            for image_file in images_folder.iterdir()
            if image_file.is_file()
        }
        updated_count = 0

        for movie in Movie.objects.all():
            expected_prefix = f"m_{movie.title}"
            matching_file = next(
                (
                    image_file
                    for image_file in image_files.values()
                    if image_file.stem == expected_prefix
                    or image_file.name == expected_prefix
                ),
                None,
            )

            if matching_file is None:
                self.stderr.write(f"Image not found for: {movie.title}")
                continue

            movie.image = os.path.join("movie", "images", matching_file.name)
            movie.save(update_fields=["image"])
            updated_count += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f"Updated image for {movie.title}: {matching_file.name}"
                )
            )

        self.stdout.write(
            self.style.SUCCESS(f"Finished updating {updated_count} movie images.")
        )
