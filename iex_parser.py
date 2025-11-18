# -*- coding: utf-8 -*-
"""
IEX binary data decoder Version 1.02

Used for three products: DEEP, DEEP+ and TOPS

DEEP+ is used to receive real-time depth of book quotations direct from IEX Exchange.
The depth of book quotations received via DEEP provide an order-by-order view of resting displayed orders.

Non-displayed orders and non-displayed portions of reserve orders are not represented in DEEP+.

DEEP+ also provides last trade price and size information. Trades resulting from
either displayed or non-displayed orders matching on IEX Exchange are reported.
Routed executions are not reported.
"""

import pandas as pd
import numpy as np
import struct
import gzip
# from itch_data_download import get_data
from tqdm import tqdm
import os
from pandas import isnull, notnull
from tabulate import tabulate
"""
String: Fixed-length ASCII byte sequence, left justified and space filled on the right

Long: 8 bytes, signed integer
Price: 8 bytes, signed integer containing a fixed-point number with 4 digits
to the right of an implied decimal point

Integer: 4 bytes, unsigned integer
Byte: 1 byte, unsigned integer
Timestamp: 8 bytes, signed integer containing a counter of nanoseconds since POSIX (Epoch) time UTC
Event Time: 4 bytes, unsigned integer containing a counter of seconds since POSIX (Epoch) time UTC

All binary fields are in little endian format.
Note that each byte is represented by two hexadecimal digits in the examples within this specification
"""

"""
System Event Message – S (0x53)
Security Directory Message – D (0x44)
Trading Status Message – H (0x48)
Retail Liquidity Indicator Message – I (0x49)
Operational Halt Status Message – O (0x4f)
Short Sale Price Test Status Message – P (0x50)
Security Event Message – E (0x45)
Add Order Message – a (0x61)
Order Modify Message – M (0x4D)
Order Delete Message – R (0x52)
Order Executed Message – L (0x4C)
Trade Message – T (0x54)
Trade Break Message – B (0x42)
Clear Book Message – C (0x43)
"""

"""
PLEASE NOTE
The Byte will transfer to hex numbers at first
can't directly use struct.unpack('<B', b'S') for example
use hex(struct.unpack('<B', b'S')) to return hex number
"""

"""
Different from the Nasdaq ITCH data, we need to deal with the message header
"""

# %% parser


def IEX_parser(path, data_product, max_read_msgs=None, dataframe=True, timestamp_conversion=True):

    if data_product not in ['DEEP', 'DEEP+', 'TOPS']:
        raise Exception(
            'Data products should be one of the following three: DEEP+, DEEP, or TOPS')

    print(f"Start processing IEX {data_product}")

    def byte_to_str(block):

        block = block.decode('ascii').rstrip()
        return block

    def byte_to_hex(block):

        block = hex(block)
        return block

    def byte_to_int(block):

        block = int.from_bytes(block, byteorder='little')
        return block

    def timestamp_conversion(block):

        block = pd.to_datetime(block, unit='ns', utc=True)
        return block

    """
    def epoch_time_convert(block):

        block = datetime.utcfromtimestamp(block/ 1_000_000_000)
        return block
    """

    """
    def iex_tp_header(msgs_blocks):

        Version 0 1 Byte 1 (0x1) Version of Transport specification
        (Reserved) 1 1 Reserved byte
        Message Protocol ID 2 2 Short Unique identifier of the higher-layer protocol
        Channel ID 4 4 Integer
        Identifies the stream of bytes/sequenced messages
        Session ID 8 4 Integer Identifies the session
        Payload Length 12 2 Short Byte length of the payload
        Message Count 14 2 Short Number of messages in the payload
        Stream Offset 16 8 Long Byte offset of the data stream
        First Message Sequence Number 24 8 Long Sequence of the first message in the segment
        Send Time 32 8 Timestamp Send time of segment

        Long: 8 bytes, signed integer
        Integer: 4 bytes, unsigned integer
        Short: 2 bytes, unsigned integer
        Byte: 1 byte unsigned integer
        Timestamp: 8 bytes, signed integer containing a counter of nanoseconds since POSIX (Epoch) time UTC


        (reserved, protocolID, channelID, sessionID,
         payload, msg_count, stream_offset, first_msg, send_time) = struct.unpack('<sHIIHHqqq', msgs_blocks)

        msgs_blocks = [reserved, protocolID, channelID, sessionID,
                       payload, msg_count, stream_offset, first_msg, send_time]

        return msgs_blocks
    """

    def SystemEvent(msgs_blocks):

        (msg_type, system_event, Time) = struct.unpack(
            '<BBq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        system_event = byte_to_hex(system_event)

        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        """

        if system_event == '0x4f':
            system_event = 'O'
        elif system_event == '0x53':
            system_event = 'S'
        elif system_event == '0x52':
            system_event = 'R'
        elif system_event == '0x4d':
            system_event = 'M'
        elif system_event == '0x45':
            system_event = 'E'
        elif system_event == '0x43':
            system_event = 'C'
        """

        msgs_blocks = {'msg_type': 'S',
                       'system_event': system_event,
                       'Time': Time
                       }

        return msgs_blocks

    def security_directory(msgs_blocks):
        """
        Message Type 0 1 Byte ‘D’ (0x44)
        Flags 1 1 Byte
        Timestamp 2 8 Timestamp
        Symbol 10 8 String
        Round Lot Size 18 4 Integer
        Adjusted POC Price 22 8 Price
        LULD Tier 30 1 Byte
        """

        (msg_type, Flags, Time, symbol, round_lot, adj_poc_price, luld_tier) = struct.unpack(
            '<BBq8sIqB', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        Falgs = byte_to_hex(Flags)
        # more details are in Appendix A of data specification

        symbol = byte_to_str(symbol)
        luld_tier = byte_to_hex(luld_tier)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {
            "msg_type": 'D',
            "Flags": Flags,
            "Time": Time,
            "symbol": symbol,
            "round_lot": round_lot,
            "adj_poc_price": adj_poc_price,
            "luld_tier": luld_tier
        }

        return msgs_blocks

    def trading_status(msgs_blocks):

        (msg_type, trd_status, Time, symbol, reason) = struct.unpack(
            '<BBq8s4s', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        symbol = byte_to_str(symbol)
        reason = byte_to_str(reason)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {'msg_type': 'H',
                       'trd_status': trd_status,
                       'Time': Time,
                       'symbol': symbol,
                       'reason': reason
                       }

        return msgs_blocks

    def retail_liquidity(msgs_blocks):

        (msg_type, retail_indicator, Time, symbol) = struct.unpack(
            '<BBq8s', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {'msg_type': 'I',
                       'retail_indicator': retail_indicator,
                       'Time': Time,
                       'symbol': symbol
                       }

        return msgs_blocks

    def operation_halt(msgs_blocks):

        (msg_type, halt_status, Time, symbol) = struct.unpack(
            '<BBq8s', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {'msg_type': 'O',
                       'halt_status': halt_status,
                       'Time': Time,
                       'symbol': symbol
                       }

        return msgs_blocks

    def short_sale_test(msgs_blocks):

        (msg_type, short_status, Time, symbol, detail) = struct.unpack(
            '<BBq8sB', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {'msg_type': 'P',
                       'short_status': short_status,
                       'Time': Time,
                       'symbol': symbol,
                       'detail': detail
                       }

        return msgs_blocks

    def security_event(msgs_blocks):

        (msg_type, security_event, Time, symbol) = struct.unpack(
            '<BBq8s', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {'msg_type': 'E',
                       'security_event': security_event,
                       'Time': Time,
                       'symbol': symbol
                       }

        return msgs_blocks

    def add_ord(msgs_blocks):

        (msg_type, side, Time, symbol, orderid, size, price) = struct.unpack(
            '<BBq8sqIq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        symbol = byte_to_str(symbol)

        msgs_blocks = {'msg_type': 'a',
                       'side': side,
                       'Time': Time,
                       'symbol': symbol,
                       'orderid': orderid,
                       'size': size,
                       'price': price
                       }

        return msgs_blocks

    def modify_ord(msgs_blocks):

        # order id will remain the same as the add order message
        # unique order id in trading session
        (msg_type, Flags, Time, symbol, orderid, size, price) = struct.unpack(
            '<BBq8sqIq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        symbol = byte_to_str(symbol)

        result = {
            "msg_type": 'M',
            "Flags": Flags,
            "Time": Time,
            "symbol": symbol,
            "orderid": orderid,
            "size": size,
            "price": price
        }

        return msgs_blocks

    def delete_ord(msgs_blocks):

        (msg_type, reserved, Time, symbol, orderid) = struct.unpack(
            '<BBq8sq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        symbol = byte_to_str(symbol)

        msgs_blocks = {
            "msg_type": 'R',
            "reserved": reserved,
            "Time": Time,
            "symbol": symbol,
            "orderid": orderid
        }

        return msgs_blocks

    def execute_ord(msgs_blocks):

        (msg_type, sale_cond, Time, symbol, orderid, size, price, tradeid) = struct.unpack(
            '<BBq8sqIqq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {
            "msg_type": 'L',
            "sale_cond": sale_cond,
            "Time": Time,
            "symbol": symbol,
            "orderid": orderid,
            "size": size,
            "price": price,
            "tradeid": tradeid
        }

        return msgs_blocks

    def trd_msg(msgs_blocks):

        (msg_type, sale_cond, Time, symbol, size, price, tradeid) = struct.unpack(
            '<BBq8sIqq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {
            "msg_type": 'T',
            "sale_cond": sale_cond,
            "Time": Time,
            "symbol": symbol,
            "size": size,
            "price": price,
            "tradeid": tradeid
        }
        return msgs_blocks

    def trd_break(msgs_blocks):

        (msg_type, sale_cond, Time, symbol, size, price, tradeid) = struct.unpack(
            '<BBq8sIqq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {
            "msg_type": 'B',
            "sale_cond": sale_cond,
            "Time": Time,
            "symbol": symbol,
            "size": size,
            "price": price,
            "tradeid": tradeid
        }
        return msgs_blocks

    def clear_book(msgs_blocks):

        (msg_type, reserved, Time, symbol) = struct.unpack(
            '<BBq8s', msgs_blocks)
        msg_type = byte_to_hex(msg_type)

        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {
            "msg_type": 'C',
            "reserved": reserved,
            "Time": Time,
            "symbol": symbol
        }
        return msgs_blocks

    def price_update(msgs_blocks):

        (msg_type, event, Time, symbol, size, price) = struct.unpack(
            '<BBq8sIq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)
        # event = byte_to_hex(event)
        msgs_blocks = {'msg_type': '8/5',
                       'event': event,
                       'Time': Time,
                       'symbol': symbol,
                       'size': size,
                       'price': price
                       }

        return msgs_blocks

    def official_price(msgs_blocks):

        (msg_type, price_type, Time, symbol, official_px) = struct.unpack(
            '<BBq8sq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {
            "msg_type": 'X',
            "price_type": price_type,
            "Time": Time,
            "symbol": symbol,
            "official_px": official_px
        }
        return msgs_blocks

    def auction_info(msgs_blocks):

        (msg_type, auction_type, Time, symbol, pairedshares, refprice, indclearingpx,
         imbshares, imbside, extnum, scheduledtime, book_clearpx, collarpx, lowerauction,
         upperauction) = struct.unpack(
            '<BBq8sIqqIBBIqqqq', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        symbol = byte_to_str(symbol)

        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {
            "msg_type": 'A',
            "auction_type": auction_type,
            "Time": Time,
            "symbol": symbol,
            "pairedshares": pairedshares,
            "refprice": refprice,
            "indclearingpx": indclearingpx,
            "imbshares": imbshares,
            "imbside": imbside,
            "extnum": extnum,
            "scheduledtime": scheduledtime,
            "book_clearpx": book_clearpx,
            "collarpx": collarpx,
            "lowerauction": lowerauction,
            "upperauction": upperauction
        }
        return msgs_blocks

    def quote_updates(msgs_blocks):

        (msg_type, flag, Time, symbol, bid_size, bid_price, ask_price, ask_size) = struct.unpack(
            '<BBq8sIqqI', msgs_blocks)
        msg_type = byte_to_hex(msg_type)
        symbol = byte_to_str(symbol)
        if timestamp_conversion:
            Time = timestamp_conversion(Time)

        msgs_blocks = {
            "msg_type": 'Q',
            "flag": flag,
            "Time": Time,
            "symbol": symbol,
            "bid_size": bid_size,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "ask_size": ask_size
        }
        return msgs_blocks

    def search_session_id(path, data_product):
        version = b'\x01'
        reserved = b'\x00'

        if data_product == "DEEP":
            protocol_id = b'\x04\x80'
        elif data_product == "DEEP+":
            protocol_id = b'\x05\x80'
        elif data_product == "TOPS":
            protocol_id = b'\x03\x80'

        channel_id = b'\x01\x00\x00\x00'

        tp_header = (version + reserved + protocol_id +
                     channel_id)

        found = False
        target_i = len(tp_header)
        i = 0
        with gzip.open(path, "rb") as f:
            try:
                cur_chunk = f.read(1)

                if not cur_chunk:
                    raise ValueError(
                        "File is empty or could not read the first chunk")

                while not found:
                    if not cur_chunk:
                        raise ValueError("No session ID found")

                    if cur_chunk[0] == tp_header[i]:
                        i += 1
                        if i == target_i:
                            found = True
                            session_id = f.read(4)
                            session_id_pos = f.tell()
                            # print(pos)
                            if not session_id:
                                raise ValueError("Session ID not found")
                            print('Session_ID found!')
                            return session_id, session_id_pos
                    else:
                        i = 0
                    cur_chunk = f.read(1)

            except:
                return None

    session_id, session_id_pos = search_session_id(
        path, data_product=data_product)

    def payload_seek(path, data_product, max_read_msgs=None):
        # allow users to set the max read

        version = b'\x01'
        reserved = b'\x00'

        if data_product == "DEEP":
            protocol_id = b'\x04\x80'
        elif data_product == "DEEP+":
            protocol_id = b'\x05\x80'
        elif data_product == "TOPS":
            protocol_id = b'\x03\x80'

        channel_id = b'\x01\x00\x00\x00'

        tp_header = (version + reserved + protocol_id +
                     channel_id + session_id)

        # hex_str = ' '.join(f'{byte:02x}' for byte in tp_header)

        payload = []
        remain_part = []

        target_i = len(tp_header)
        i = 0

        if isnull(max_read_msgs):
            print(
                'maximum number of messages read does not provide. Read the whole file by default')
            max_read = os.path.getsize(path)
            print(f'Total bytes: {max_read}')

        else:
            max_read = max_read_msgs
            print(f'Read maximum number of messages {max_read_msgs}')

        with gzip.open(path, "rb") as f:
            cur_chunk = f.read(1)
            if isnull(max_read_msgs):
                read = 1
            else:
                read = 0

            if notnull(max_read_msgs):
                unit = 'msgs'
            else:
                unit = 'bytes'

            with tqdm(total=max_read, desc="Reading", ncols=100, unit=unit) as pbar:

                while cur_chunk and read < max_read:
                    # print(i)

                    if cur_chunk[0] == tp_header[i]:
                        i += 1
                        if i == target_i:

                            remaining_header = struct.unpack(
                                '<HHqqqH', f.read(30))

                            cur_msg_payload_len = remaining_header[0]
                            messages_count = remaining_header[1]
                            cur_stream_offset = remaining_header[2]
                            sfirst_sequence_number = remaining_header[3]
                            cur_send_time = remaining_header[4]
                            msg_length = remaining_header[5]
                            # pos= f.tell()

                            if msg_length != 0:
                                # recording the payload (message blocks)
                                msgs_data = f.read(msg_length)
                                payload.append(msgs_data)
                                pbar.set_postfix_str(
                                    f'Payloads: {len(payload)}')
                                # since we use tqdm for progress, so you can't use print directly
                                # do not print in new lines

                                if notnull(max_read_msgs):
                                    read += 1
                                    pbar.update(1)
                                else:
                                    read += msg_length
                                    pbar.update(msg_length)

                            i = 0  # remember to reset i to 0
                    else:
                        i = 0

                    if isnull(max_read_msgs):
                        pbar.update(1)
                        read += 1
                    cur_chunk = f.read(1)

        return payload

    payload = payload_seek(path, data_product=data_product,
                           max_read_msgs=max_read_msgs)

    # f.read() can directly read from the current point, no need to
    # explicitly specify the starting point
    # TODO: need to gove the file path
    # TODO: Need to use multiprocessing
    # TODO: This version only supports local file

    msgs_SystemEvent = []
    msgs_security_directory = []
    msgs_trading_status = []
    msgs_retail_liquidity = []
    msgs_operation_halt = []
    msgs_short_sale_test = []
    msgs_security_event = []
    msgs_add_ord = []
    msgs_modify_ord = []
    msgs_delete_ord = []
    msgs_execute_ord = []
    msgs_trd_msg = []
    msgs_trd_break = []
    msgs_clear_book = []
    msgs_price_update = []
    msgs_official_price = []
    msgs_auction_info = []
    msgs_quote_updates = []

    # initialize the progress bar
    # cutting the binary data to blocks
    # Remember: In .read(n) from a binary file,
    # n is the number of bytes to read starting from the current file pointer
    # it does NOT include the current position itself.
    # No msg field needed in msg function
    # TODO: check the length

    # TODO: Note that the message type in IEX is Byte (1 byte unsinged integer)
    # This will be shown as the hex number at first, I will transfer to integer
    # So I will want to know the message header in binary format at first
    # then start to proces
    # This is all consistent to what the specification file shows

    # TODO: read the header
    # one might not read the standard pcap data'
    # print(start)

    element = 0

    with tqdm(total=len(payload), desc="Reading", ncols=100, unit='payload') as pbar:

        while element < len(payload):
            # print(element)
            msg_data = payload[element]
            msg_header = msg_data[:1]

            if msg_header == b'S':
                msgs_SystemEvent.append(msg_data)

            elif msg_header == b'D':
                msgs_security_directory.append(msg_data)

            elif msg_header == b'H':
                msgs_trading_status.append(msg_data)

            elif msg_header == b'I':
                msgs_retail_liquidity.append(msg_data)

            elif msg_header == b'O':
                msgs_operation_halt.append(msg_data)

            elif msg_header == b'P':
                msgs_short_sale_test.append(msg_data)

            elif msg_header == b'E':
                msgs_security_event.append(msg_data)

            elif msg_header == b'a':
                msgs_add_ord.append(msg_data)

            elif msg_header == b'M':
                msgs_modify_ord.append(msg_data)

            elif msg_header == b'R':
                msgs_delete_ord.append(msg_data)

            elif msg_header == b'L':
                msgs_execute_ord.append(msg_data)

            elif msg_header == b'T':
                msgs_trd_msg.append(msg_data)

            elif msg_header == b'B':
                msgs_trd_break.append(msg_data)

            elif msg_header == b'C':
                msgs_clear_book.append(msg_data)

            elif ((msg_header == b'8') | (msg_header == b'5')) and data_product == "DEEP":
                msgs_price_update.append(msg_data)

            elif (msg_header == b'X') and (data_product == "DEEP" | data_product == "DEEP"):
                msgs_official_price.append(msg_data)

            elif msg_header == b'A' and (data_product == "TOPS" | data_product == "DEEP"):
                msgs_auction_info.append(msg_data)

            elif msg_header == b'Q' and (data_product == "TOPS"):
                msgs_quote_updates.append(msg_data)

            element += 1
            pbar.update(1)

    msgs_SystemEvent = list(
        map(lambda item: SystemEvent(item), msgs_SystemEvent))
    msgs_security_directory = list(
        map(lambda item: security_directory(item), msgs_security_directory))
    msgs_trading_status = list(
        map(lambda item: trading_status(item), msgs_trading_status))
    msgs_retail_liquidity = list(
        map(lambda item: retail_liquidity(item), msgs_retail_liquidity))
    msgs_operation_halt = list(
        map(lambda item: operation_halt(item), msgs_operation_halt))
    msgs_short_sale_test = list(
        map(lambda item: short_sale_test(item), msgs_short_sale_test))
    msgs_security_event = list(
        map(lambda item: security_event(item), msgs_security_event))
    msgs_add_ord = list(
        map(lambda item: add_ord(item), msgs_add_ord))
    msgs_modify_ord = list(
        map(lambda item: modify_ord(item), msgs_modify_ord))
    msgs_delete_ord = list(
        map(lambda item: delete_ord(item), msgs_delete_ord))
    msgs_execute_ord = list(
        map(lambda item: execute_ord(item), msgs_execute_ord))
    msgs_trd_msg = list(
        map(lambda item: trd_msg(item), msgs_trd_msg))
    msgs_trd_break = list(
        map(lambda item: trd_break(item), msgs_trd_break))
    msgs_clear_book = list(
        map(lambda item: clear_book(item), msgs_clear_book))
    msgs_price_update = list(
        map(lambda item: price_update(item), msgs_price_update))
    msgs_official_price = list(
        map(lambda item: official_price(item), msgs_official_price))
    msgs_auction_info = list(
        map(lambda item: auction_info(item), msgs_auction_info))
    msgs_quote_updates = list(
        map(lambda item: quote_updates(item), msgs_quote_updates))

    results = {'System_Event': msgs_SystemEvent,
               'security_directory': msgs_security_directory,
               'trading_status': msgs_trading_status,
               'retail_liquidity': msgs_retail_liquidity,
               'operation_halt': msgs_operation_halt,
               'short_sale_test': msgs_short_sale_test,
               'execute_ord': msgs_execute_ord,
               'add_ord': msgs_add_ord,
               'modify_ord': msgs_modify_ord,
               'delete_ord': msgs_delete_ord,
               'trd_msg': msgs_trd_msg,
               'trd_break': msgs_trd_break,
               'clear_book': msgs_clear_book,
               'price_update': msgs_price_update,
               'official_price': msgs_official_price,
               'auction_info': msgs_auction_info,
               'quote_updates': msgs_quote_updates}

    results_info = pd.DataFrame({'msg_types': list(results.keys()), 'number_msgs': list(
        map(lambda item: len(item), results.values()))})

    print(tabulate(results_info, headers=[
        'msg_types', 'number_msgs'], tablefmt="grid"))

    if dataframe == True:

        results = dict(
            map(lambda item: (item[0], pd.DataFrame(item[1])), results.items()))

    return results



