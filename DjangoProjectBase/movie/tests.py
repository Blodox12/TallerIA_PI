from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from django.test import TestCase

from .models import Movie


class RecommendationViewTests(TestCase):
	def setUp(self):
		Movie.objects.create(
			title='Less similar',
			description='A movie',
			emb=np.array([0.0, 1.0], dtype=np.float32).tobytes(),
		)
		Movie.objects.create(
			title='Most similar',
			description='Another movie',
			emb=np.array([1.0, 0.0], dtype=np.float32).tobytes(),
		)

	def test_recommendation_returns_movie_with_highest_similarity(self):
		response = SimpleNamespace(
			data=[SimpleNamespace(embedding=[1.0, 0.0])],
		)
		client = SimpleNamespace(
			embeddings=SimpleNamespace(create=lambda **kwargs: response),
		)

		with patch.dict('os.environ', {'openai_apikey': 'test-key'}), patch(
			'movie.views.OpenAI', return_value=client
		):
			page = self.client.post(
				'/recommendation/',
				{'prompt': 'an action movie'},
			)

		self.assertEqual(page.status_code, 200)
		self.assertContains(page, 'Most similar')
		self.assertContains(page, 'Similarity: 1.0000')

# Create your tests here.
