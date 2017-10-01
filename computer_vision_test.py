"""
Testing methods for ImageSet. Primary behavior curently tested 
is the verification of valid/invalid image collections.
"""

import unittest
import os
import shutil
from computer_vision import ImageSet, TransferModel
from PIL import Image

# TODO:
# - Add tests for copy_dir(...), create_category_dir_shell(...), and 
# partition_image_collection(...)
class TestImageSet(unittest.TestCase):
    """
    Tests with the purpose of verifying the behavior of ImageSet.
    """

    @staticmethod
    def create_image(path, dims=(150, 150)):
        img = Image.new('RGB', dims)
        img.save(path)

    def setUp(self):
        """
        Construct mock valid and invalid image collections.
        """
        
        # Construct valid Standard Image Collection
        os.makedirs("Valid")
        os.makedirs("Valid/Class1")
        self.create_image("Valid/Class1/img1.jpg")
        self.create_image("Valid/Class1/img2.jpg")
        os.makedirs("Valid/Class2")
        self.create_image("Valid/Class2/img1.jpg")
        self.create_image("Valid/Class2/img2.jpg")

        # Construct image collection without classes
        os.makedirs("NoLabelData")
        self.create_image("NoLabelData/img1.jpg")
        self.create_image("NoLabelData/img2.jpg")
        
        # Construct image collection with a single class
        os.makedirs("OneLabelData")
        os.makedirs("OneLabelData/Class1")
        self.create_image("OneLabelData/Class1/img1.jpg")
        self.create_image("OneLabelData/Class1/img2.jpg")
        
        # Construct single image file
        self.create_image("singleImg.jpg")

    def tearDown(self):
        """
        Remove mock image data.
        """
        shutil.rmtree("Valid")
        shutil.rmtree("NoLabelData")
        shutil.rmtree("OneLabelData")
        os.remove("singleImg.jpg")

    def test_partition_list(self):

        test_list = [1, 2, 3, 4]
        small_out, large_out = ImageSet.partiton_list(test_list, 2)
        self.assertEqual(len(small_out), 2)
        self.assertEqual(len(large_out), 2)

        test_list = [1, 2, 3, 4]
        small_out, large_out = ImageSet.partiton_list(test_list, 0)
        self.assertEqual(len(small_out), 0)
        self.assertEqual(len(large_out), 4)

    def test_get_collection_info(self):
        """
        Verifies that get_collection_info(...) can correctly summarize a
        Standard Image Collection.
        """

        # Test valid collections
        self.assertEqual(ImageSet.get_collection_info("Valid"), (2, 4))

        # Test invalid collections
        self.assertIs(ImageSet.get_collection_info("singleImg.jpg"), None)
        self.assertIs(ImageSet.get_collection_info("NonLabelledData"), None)
        self.assertIs(ImageSet.get_collection_info("OneLabelData"), None)

    def test_is_collection_valid(self):
        """
        Verifies that is_collection_valid(...) can correctly identify a
        Standard Image Collection.
        """

        # Test valid collections
        self.assertTrue(ImageSet.is_collection_valid("Valid"))

        # Test invalid collections
        self.assertFalse(ImageSet.is_collection_valid("singleImg.jpg"))
        self.assertFalse(ImageSet.is_collection_valid("NonLabelledData"))
        self.assertFalse(ImageSet.is_collection_valid("OneLabelData"))

    def test_ImageSet_construction(self):
        """
        Verifies that ImageSet construction cannot be done with invalid
        collections.
        """

        # Test construction with invalid collections
        with self.assertRaises(AttributeError):
            ImageSet("singleImg.jpg")
        with self.assertRaises(AttributeError):
            ImageSet("NonLabelledData")
        with self.assertRaises(AttributeError):
            ImageSet("OneLabelData")

if __name__ == "__main__":
    unittest.main()
