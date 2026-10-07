"""Protect the exact first-frame and no-seed contract for gateway requests."""
import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from generation.backends import Seedance25Video


class SeedanceGatewayTests(unittest.TestCase):
    def test_gateway_json_preserves_image_and_uses_provider_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'first_frame.png'
            image.write_bytes(b'exact supplied image')
            model = Seedance25Video(base_url='https://provider.example/v1', request_format='litellm_json')
            body = model._request_body(first_frame=str(image), prompt='Unmodified prompt.', num_frames=120, seed=42)
            self.assertEqual(body['prompt'], 'Unmodified prompt.')
            self.assertNotIn('seed', body)
            self.assertNotIn('seconds', body)
            self.assertEqual(body['extra_body']['duration'], 5)
            encoded = body['extra_body']['frame_images'][0]['image_url']['url'].split(',', 1)[1]
            self.assertEqual(base64.b64decode(encoded), image.read_bytes())
            self.assertEqual(model._join_url('/v1/videos'), 'https://provider.example/v1/videos')

    def test_gateway_download_uses_original_routed_job_id(self):
        model = Seedance25Video(base_url='https://provider.example/v1', request_format='litellm_json')
        with patch.object(model, '_download_via_file_id', return_value=(123, 'downloaded')) as download:
            model._download_result({'id': 'original-routed-id', 'url': 'https://unrelated.example/video.mp4'}, 'out.mp4')
            download.assert_called_once_with('original-routed-id', 'out.mp4')


if __name__ == '__main__':
    unittest.main()
