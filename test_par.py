from __future__ import print_function

import unittest
import os
import numpy
import traceback

from HSTB.drivers import par
from HSTP.Pydro import par as par_old


class TestPAR(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        datapath = os.path.join(os.path.dirname(os.path.dirname(par.__file__)), "demos", "Hydrographic_Python", "Bathymetry", "Kongsberg")
        filename = os.path.join(datapath, r"0008_20161016_184328_S222_EM2040.all")
        self.infile = par.allRead(filename)
        self.infile_old = par_old.allRead(filename)
        print("Mapping the file, this can be slow -- please wait...\n")
        self.infile.mapfile(show_progress=False)  # find where the data packets are and keep track so we can read them quickly later
        self.infile_old.mapfile()  # find where the data packets are and keep track so we can read them quickly later

    def tearDown(self):
        pass

    def jiggle(self, data, val):
        """Modify the numeric data of any BaseData derived class"""
        for n, name in enumerate(data.header.dtype.names):
            if data.header[name].dtype.kind in ("f", "i"):
                data.header[name] += val

    def load_data(self, pkt):
        """Get the datagram instances for the given pkt id and return the NEW style data and OLD style data"""
        data = self.infile.getrecord(pkt, 0)
        data_oldstyle = self.infile_old.getrecord(pkt, 0)
        if data_oldstyle is None or data is None:
            self.skipTest("Did not find data for pkt {}".format(pkt))
        return data, data_oldstyle

    def show_diff(self, h1, h2, tol):
        """Print out the fields that don't match between two numpy data structures and their respective values
        and also specify which fields did match"""
        if h1.shape != h2.shape:
            print("Shapes of the data did not match -- records are missing or extra")
            print(h1.shape, "vs", h2.shape)
            print("Probably a datagram length mismatch or end of record characters being misinterpreted")
        else:
            fields = list(h1.dtype.names)
            print("The following fields FAILED to match:")
            for fld in h1.dtype.names:
                if (h1[fld] != h2[fld]).any():
                    try:
                        if numpy.fabs(h1[fld] - h2[fld]).max() > tol:
                            failed = True
                        else:
                            failed = False
                    except:
                        failed = True
                    if failed:
                        print(fld)
                        try:
                            print("Max Diff:", numpy.fabs(h1[fld] - h2[fld]).max())
                        except:
                            pass
                        print(h1[fld], h2[fld])
                        fields.remove(fld)
            print("The following fields matched:\n",fields)

    def compare_headers(self, h1, h2, tol=0.0):
        """Compare two numpy data structs and allow numeric fields to mismatch 
        given a certain tolerance to account for roundoff"""
        if h1.shape != h2.shape:
            return False
        if (h1 != h2).any():
            for fld in h1.dtype.names:
                if (h1[fld] != h2[fld]).any():
                    try:
                        if numpy.fabs(h1[fld] - h2[fld]).max() > tol:
                            return False
                    except:
                        return False
        return True
    
    def equality_test(self, data, ref_data, msg="New class result doesn't match old class", tol=0.0):
        """Compares a BaseData derived 'data' with a numpy array having matching field names.
        Causes a failed test if the data does not match within tolerance""" 
        if not self.compare_headers(data.header, ref_data, tol):
            print("\n", data.__class__)
            self.show_diff(data.header, ref_data, tol)
            self.fail(msg+" for packet type {}".format(data.__class__))
    
    def writable_test(self, data, args=[], tol=0.0):
        """Given a datagram instance derived from BaseData,
        Try to modify the values and write it out to memory then read it back in 
        and compare to the original data.  There is some issue with roundoff currently so a 
        tolerance can be specified."""
        datablock = data.get_datablock()  # get a buffer string that could be written back to a file
        data_cloned = data.__class__(datablock, *args)  # read and print the data to see if it worked
        self.equality_test(data, data_cloned.header, "writing then reading back the data doesn't match original data", tol)
        self.jiggle(data_cloned, 1)
        datablock = data_cloned.get_datablock()  # get a buffer string that could be written back to a file
        data_jiggled = data.__class__(datablock, *args)  # read and print the data to see if it worked
        self.jiggle(data_jiggled, -1)
        datablock = data_jiggled.get_datablock()  # get a buffer string that could be written back to a file
        data_back_to_original = data.__class__(datablock, *args)  # read and print the data to see if it worked
        self.equality_test(data, data_back_to_original.header, "modifying, writing, reading then reverting back the data doesn't match original data", tol)
        return data_back_to_original

    def read_test(self, pkt, tol=0.0):
        """Load the packet data using the original par routines and the revised base class derived ones.
        Compare the results and return the two datagram instances (new, old)"""
        data, data_oldstyle = self.load_data(pkt)
        self.equality_test(data, data_oldstyle.header, tol=tol)
        return data, data_oldstyle
        
    def test_48_read(self):
        self.read_test("48")

    def test_48_write(self):
        data, data_oldstyle = self.load_data("48")
        self.writable_test(data)

    def test_49_read(self):
        self.read_test("49")

    def test_49_write(self):
        data, data_oldstyle = self.load_data("49")
        self.writable_test(data, tol = 0.03)  # there is an issue with the units conversion and back that makes a roundoff error

    def test_51_read(self):
        data, data_oldstyle = self.read_test("51")
        try:
            data_oldstyle.data
        except AttributeError:
            pass  # was a content type that doesn't create data
        else:
            self.assertEqual(data.data, data_oldstyle.data, "writing then reading back the data doesn't match original data for packet type {}".format(data.__class__))

    def test_51_write(self):
        data, data_oldstyle = self.load_data("51")
        data_back_to_original = self.writable_test(data, args=[data.model])
        try:
            data_oldstyle.data
        except AttributeError:
            pass  # was a content type that doesn't create data
        else:
            self.assertEqual(data.data, data_back_to_original.data, "writing then reading back the data doesn't match original data for packet type {}".format(data.__class__))

    def test_65_read(self):
        data, data_oldstyle = self.read_test("65")
        self.equality_test(data.att, data_oldstyle.data)  # compare the attitude sub-data

    def test_65_write(self):
        data, data_oldstyle = self.load_data("65")
        data_back_to_original = self.writable_test(data, args=[data.time])
        self.equality_test(data.att, data_back_to_original.att.header, "Sub-data didn't get preserved", tol=0.03)
        self.writable_test(data.att, args=[data.time], tol=0.03) # compare the attitude data as well

    def test_66_read(self):
        data, data_oldstyle = self.read_test("66")
        self.assertEqual(data.raw_data, data_oldstyle.raw_data)
        self.assertEqual(data.raw_data, data_back_to_original.raw_data)
    def test_66_write(self):
        data, data_oldstyle = self.load_data("66")
        data_back_to_original = self.writable_test(data)

    def test_67_read(self):
        data, data_oldstyle = self.read_test("67")
    def test_67_write(self):
        data, data_oldstyle = self.load_data("67")
        data_back_to_original = self.writable_test(data)

    def test_68_read(self):
        data, data_oldstyle = self.read_test("68")
        self.equality_test(data.xyz, data_oldstyle.data)  # compare the xyz sub-data
    def test_68_write(self):
        data, data_oldstyle = self.load_data("68")
        data_back_to_original = self.writable_test(data, args=[data.header])
        self.equality_test(data.xyz, data_back_to_original.xyz.header, "Sub-data didn't get preserved")
        self.writable_test(data.xyz, args=[data.header]) # compare the data after modifications as well

    def test_71_read(self):
        data, data_oldstyle = self.read_test("71")
        self.equality_test(data.ss, data_oldstyle.data)
    def test_71_write(self):
        data, _data_oldstyle = self.load_data("71")
        data_back_to_original = self.writable_test(data, args=[data.POSIXtime])
        self.equality_test(data.ss, data_back_to_original.ss.header, "Sub-data didn't get preserved")
        self.writable_test(data.ss, args=[data.POSIXtime])

    def test_73_read(self):
        data, data_oldstyle = self.read_test("73")
        self.assertEqual(data.settings, data_oldstyle.settings)
    def test_73_write(self):
        data, _data_oldstyle = self.load_data("73")
        data_back_to_original = self.writable_test(data)
        self.assertEqual(data.settings, data_back_to_original.settings)

    def test_78_read(self):
        data, data_oldstyle = self.read_test("78")
        self.equality_test(data.tx_data, data_oldstyle.tx)
        self.equality_test(data.rx_data, data_oldstyle.rx)
    def test_78_write(self):
        data, _data_oldstyle = self.load_data("78")
        data_back_to_original = self.writable_test(data, args=[data.pingtime])
        self.writable_test(data.tx_data, tol=0.001)
        self.writable_test(data.rx_data, tol=0.2)

    def test_79_read(self):
        data, data_oldstyle = self.read_test("79")
        self.assertTrue((data.data==data_oldstyle.data).all())
    def test_79_write(self):
        data, _data_oldstyle = self.load_data("79")
        data_back_to_original = self.writable_test(data)
        self.assertEqual(data.data, data_back_to_original.data)

    def test_80_read(self):
        data, data_oldstyle = self.read_test("80")
        data.parse_raw()  # have to force class to parse the data
        data_oldstyle.parse_raw()  # have to force class to parse the data
        self.equality_test(data.gg_data, data_oldstyle.source_data)
    def test_80_write(self):
        data, _data_oldstyle = self.load_data("80")
        data.parse_raw()  # have to force class to parse the data
        data_back_to_original = self.writable_test(data)
        data_back_to_original.parse_raw()  # have to force class to parse the data
        self.equality_test(data.gg_data, data_back_to_original.gg_data.header, "Sub-data didn't get preserved")
        self.writable_test(data.gg_data)

    def test_82_read(self):
        data, data_oldstyle = self.read_test("82")
    def test_82_write(self):
        data, _data_oldstyle = self.load_data("82")
        data_back_to_original = self.writable_test(data)

    def test_83_read(self):
        data, data_oldstyle = self.read_test("83")
        self.assertEqual(data.samples, data_oldstyle.samples)
    def test_83_write(self):
        data, _data_oldstyle = self.load_data("83")
        data_back_to_original = self.writable_test(data)
        self.assertEqual(data.samples, data_back_to_original.samples)

    def test_85_read(self):
        data, data_oldstyle = self.read_test("85")
        self.equality_test(data.ss, data_oldstyle.data)
    def test_85_write(self):
        data, _data_oldstyle = self.load_data("85")
        data_back_to_original = self.writable_test(data)
        self.equality_test(data.ss, data_back_to_original.ss.header, "Sub-data didn't get preserved", tol=0.03)
        self.writable_test(data.ss, args=[data.header['DepthResolution'] * 0.01], tol=0.03)

    def test_88_read(self):
        data, data_oldstyle = self.read_test("88")
        self.equality_test(data.xyz, data_oldstyle.data)
    def test_88_write(self):
        data, _data_oldstyle = self.load_data("88")
        data_back_to_original = self.writable_test(data)
        self.equality_test(data.xyz, data_back_to_original.xyz.header, "Sub-data didn't get preserved", tol=0.3)
        self.writable_test(data.xyz, tol=0.3)  # IncidenceAngleAdjustment has issues

    def test_89_read(self):
        data, data_oldstyle = self.read_test("89")
        self.equality_test(data.beaminfo_data, data_oldstyle.beaminfo)
        # self.equality_test(data.samples_data, data_oldstyle.samples)
        self.assertTrue((data.samples == data_oldstyle.samples).all())  # the old style changed the array from structured array to plain float16
    def test_89_write(self):
        data, _data_oldstyle = self.load_data("89")
        data_back_to_original = self.writable_test(data)
        self.equality_test(data.beaminfo_data, data_back_to_original.beaminfo_data.header, "Sub-data didn't get preserved")
        self.equality_test(data.samples_data, data_back_to_original.samples_data.header, "Sub-data didn't get preserved", tol=0.11)
        self.writable_test(data.beaminfo_data)
        self.writable_test(data.samples_data, tol=0.22)  # hex to float conversion for samples leads to roundoff each time

    def test_102_read(self):
        data, data_oldstyle = self.read_test("102")
        self.equality_test(data.tx_data, data_oldstyle.tx)
        self.equality_test(data.rx_data, data_oldstyle.rx)
    def test_102_write(self):
        data, data_oldstyle = self.load_data("102")
        data_back_to_original = self.writable_test(data)
        self.equality_test(data.tx_data, data_back_to_original.tx_data.header, "Sub-data didn't get preserved")
        self.equality_test(data.rx_data, data_back_to_original.rx_data.header, "Sub-data didn't get preserved")
        self.writable_test(data.tx_data)
        self.writable_test(data.rx_data)

    def test_104_read(self):
        data, data_oldstyle = self.read_test("104")
    def test_104_write(self):
        data, _data_oldstyle = self.load_data("104")
        data_back_to_original = self.writable_test(data)

    def test_107_read(self):
        data, data_oldstyle = self.read_test("107")
        self.equality_test(data.tx_data, data_oldstyle.tx)
        self.equality_test(data.rx_data, data_oldstyle.rx)
    def test_107_write(self):
        data, _data_oldstyle = self.load_data("107")
        data_back_to_original = self.writable_test(data)
        self.equality_test(data.tx_data, data_back_to_original.tx_data.header, "Sub-data didn't get preserved")
        self.equality_test(data.rx_data, data_back_to_original.rx_data.header, "Sub-data didn't get preserved")
        self.writable_test(data.tx_data)
        self.writable_test(data.rx_data)

    def test_109_read(self):
        data, data_oldstyle = self.read_test("109")
        self.assertEqual(data.data, data_oldstyle.data)
    def test_109_write(self):
        data, _data_oldstyle = self.load_data("109")
        data_back_to_original = self.writable_test(data)
        self.assertEqual(data.data, data_back_to_original.data)

    def test_110_read(self):
        data, data_oldstyle = self.read_test("110")
        try:
            data.att_data.raw_data
        except:
            self.assertTrue((data.source_data==data_oldstyle.source_data).all())
        else:
            self.equality_test(data.att_data.raw_data, data_oldstyle.source_data)
    def test_110_write(self):
        data, _data_oldstyle = self.load_data("110")
        data_back_to_original = self.writable_test(data, args=[data.time])
        self.equality_test(data.raw_data, data_back_to_original.raw_data.header, "Sub-data didn't get preserved")
        self.writable_test(data.raw_data)


if "__main__" == __name__:
    reload(par)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPAR)
    unittest.TextTestRunner(verbosity=2).run(suite)
