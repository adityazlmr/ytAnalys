from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


def preprocess_comments(comment):

    if not isinstance(comment, str):
        return ''
    
    # Case folding
    comment = comment.lower()

    # Cleansing
    comment = re.sub(r'<.*?>', '', comment)  # Menghapus semua tag HTML
    comment = re.sub(r'http\S+|www\S+', '', comment)  # Menghapus URL atau tautan web
    comment = re.sub(r'(\w+\d+\w+)', lambda x: x.group().replace('', ' '), comment)  # Menghapus karakter dari kata yang berisi kombinasi huruf, angka, dan huruf lain
    comment = re.sub(r'\W', ' ', comment)  # Mengganti karakter non-huruf dan non-angka menjadi spasi
    comment = re.sub(r'[^A-Za-z0-9\s]', '', comment)  # Menghapus karakter selain huruf, angka, dan spasi
    comment = re.sub(r'[^\x00-\x7F]+', '', comment)  # Menghapus karakter non-ASCII
    comment = re.sub(r'[^\w\s]', '', comment)  # Menghapus karakter non-huruf, non-angka, dan non-spasi

    # Tokenizing (using NLTK)
    words = word_tokenize(comment)

    # Stop words removal using predefined Sastrawi stop words for Bahasa Indonesia
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    words = stopword_remover.remove(' '.join(words)).split()

    # Manual Stop words removal
    manual_stopwords = {'terasa','wkwkwkwk', 'degenga', 'naahh', 'an', 'ke', 'pp', 'lainnya', 'diri', 'kala', 'tetapi', 'tapi', 'setibanya', 
                        'bila', 'lewat', 'sajalah', 'bagi', 'seluruh', 'sesudah', 'tunjuk', 'tersebutlah', 'siap', 'jangan', 'mf','nih','xg','kn', 'loh',
                        'tuturnya', 'sebagai', 'setempat', 'pihak', 'seterusnya', 'guna', 'tepat', 'sebegitu', 'yang', 'kann', 'kalau','klo','door','kl',
                        'pertama', 'itu', 'Hallo', 'mimimg', 'Dan','bs','laaaaah', 'yg', 'atw', 'wkwkwkkw', 'wkwkwkwk', 'amp', 'gmana', 'heee','dooor',
                        'bahkan','sm', 'sebentar', 'az', 'tm', 'jd', 'pgn', 'gw', 'sudah', 'tuh', 'pdhl', 'wkwk', 'sndri', 'sih', 'af', 'gmn', 'klau','kalo',
                        'msh','tuwekgoblokcokelek', 'bimsalabim', 'cukpng','atsupun', 'sabux', 'soalx', 'mngkin', 'ngk', 'ni', 'tddk', 'aplagi', 'sus',
                        'mj', 'mak', 'burem', 'yqng', 'neng', 'bangetu','wkwkkw','maaeen','dah', 'mangka', 'donk', 'bg','dri', 'ngga', 'kk', 'au', 
                        'lo', 'cmn', 'tu', 'hahaha', 'wkwkwkw','wkwkwkwkw','woww','hhaahha', 'wkwkw','haha','hahahaha','wih', 'ntr', 'wooo', 'allah',
                        'dll','ooooooh','bia','sa', 'mk','tuk', 'quot', 'hmmm', 'wk', 'ny'}
    
    words = [word for word in words if word not in manual_stopwords]

    # Normalization
    normalization = {
        r'\bga\b': 'tidak',
        r'\btak\b': 'tidak',
        r'\bg\b': 'tidak',
        r'\boknum\b': 'oknum',
        r'\bdinego\b': 'ditawar',
        r'\bdimusnahkn\b': 'dimusnahkan',
        r'\bdbersm\b': 'bersama',
        r'\bbkl\b': 'akan',
        r'\bsebabagian\b': 'sebagian',
        r'\bkepolisin\b': 'kepolisian',
        r'\bnego\b': 'tawar',
        r'\bsy\b': 'saya',
        r'\btrs\b': 'terus',
        r'\bkelar\b': 'selesai',
        r'\bbesas\b': 'besar',
        r'\bmn\b': 'mana',
        r'\bmlempem\b': 'lemah',
        r'\bmw\b': 'mau',
        r'\bth\b': 'tahun',
        r'\bmenghikum\b': 'menghukum',
        r'\bgk\b': 'tidak',
        r'\bdg\b': 'dengan',
        r'\bbs\b': 'bisa',
        r'\bjd\b': 'jadi',
        r'\blg\b': 'lagi',
        r'\budah\b': 'sudah',
        r'\bcri\b': 'cari',
        r'\bklu\b': 'kalau',
        r'\bbru\b': 'baru',
        r'\bmo\b': 'mau',
        r'\bjg\b': 'juga',
        r'\bsj\b': 'saja',
        r'\bdn\b': 'dan',
        r'\bad\b': 'ada',
        r'\bsya\b': 'saya',
        r'\bbsa\b': 'bisa',
        r'\bshg\b': 'sehingga',
        r'\butk\b': 'untuk',
        r'\bkrn\b': 'karena',
        r'\bjgn\b': 'jangan',
        r'\bbgt\b': 'sangat',
        r'\btrus\b': 'terus',
        r'\booknum\b':'oknum',
        r'\bbrni\b':'berani',
        r'\bknum\b':'oknum',
        r'\brakya\b':'rakyat',
        r'\bbgtu\b':'begitu',
        r'\blucuuu\b':'lucu',
        r'\bmbukak\b':'buka',
        r'\bsampe\b': 'sampai',
        r'\bbrantas\b':'berantas',
        r'\banghota\b':'anggota',
        r'\bpredi\b':'fredy',
        r'\bbuktix\b':'bukti',
        r'\bskrang\b':'sekarang',
        r'\bnarkoboy\b':'narkoba',
        r'\bwlopun\b':'walaupun',
        r'\bbngga\b':'bangga',
        r'\bbnyak\b':'banyak',
        r'\bngerasa\b':'rasa',
        r'\bdngn\b':'dengan',
        r'\bpnankapan\b':'nampak',
        r'\bsbesar\b':'sebesar',
        r'\bkrna\b':'karena',
        r'\bslama\b':'selama',
        r'\bblakang\b':'belakang',
        r'\btngan\b':'tangan',
        r'\bkarna\b':'karena',
        r'\bmsuk\b':'masuk',
        r'\bilang\b': 'hilang',
        r'\bdoyan\b':'suka',
        r'\bketangkeo\b':'tangkap',
        r'\bhnaya\b':'hanya',
        r'\bexsekusi\b':'eksekusi',
        r'\btrima\b':'terima',
        r'\bmngali\b':'gali',
        r'\bberbagasa\b':'bahasa',
        r'\bhkum\b':'hukum',
        r'\btrtangkap\b':'tangkap',
        r'\bgnti\b':'gnti',
        r'\btrpilih\b':'terpilih',
        r'\bmnjadi\b':'menjadi',
        r'\bmembedakn\b':'beda',
        r'\btdak\b':'tidak',
        r'\bsmua\b': 'semua',
        r'\bjga\b': 'juga',
        r'\bsya\b': 'saya',
        r'\borg\b': 'orang',
        r'\bksh\b': 'kasih',
        r'\bmanpaat\b': 'manfaat',
        r'\bthn\b': 'tahun',
        r'\bskrng\b': 'sekarang',
        r'\bisilop\b': 'polisi',
        r'\btrtntu\b': 'tentu',
        r'\brsiko\b': 'resiko',
        r'\bknpa\b': 'kenapa',
        r'\bmrugikn\b':'rugi',
        r'\bmnggunak\b': 'guna',
        r'\brsiko\b': 'resiko',
        r'\btrlibat\b':'libat',
        r'\bbrpngaruh\b':'pengaruh',
        r'\bsmpe\b':'sampai',
        r'\bmask\b':'masuk',
        r'\bpsti\b':'pasti',
        r'\bbrtshun\b':'tahun',
        r'\btrbukti\b': 'bukti',
        r'\bmnggunakn\b': 'guna',
        r'\bngeriiiiii\b': 'ngeri',
        r'\bkmatia\b': 'mati',
        r'\bbrpngsruh\b':'pengaruh',
        r'\bbsar\b':'besar',
        r'\bbrti\b': 'arti',
        r'\bbnyk\b': 'banyak',
        r'\bdgan\b': 'dengan',
        r'\buanga\b': 'uang',
        r'\bkyak\b': 'seperti',
        r'\bkmna\b': 'kemana',
        r'\btdk\b': 'tidak',
        r'\blawak\b':'lucu',
        r'\bjls\b': 'jelas',
        r'\bminggat\b': 'pergi',
        r'\bfibodohi\b': 'bodoh',
        r'\bmenguwasai\b': 'kuasa',
        r'\bnarkiba\b':'narkoba',
        r'\bnarcoba\b':'narkoba',
        r'\bskrg\b': 'sekarang',
        r'\bcman\b': 'hanya',
        r'\baktingya\b':'akting',
        r'\bmwmusnahkn\b': 'musnah',
        r'\bkalaw\b': 'kalau',
        r'\bbnr\b': 'benar',
        r'\bemng\b':'memang',
        r'\bbakalan\b': 'akan',
        r'\btankaap\b': 'tangkap',
        r'\bprcaya\b': 'percaya',
        r'\bjngn\b': 'jangan',
        r'\blgi\b': 'lagi',
        r'\bnnti\b': 'nanti',
        r'\bgue\b': 'saya',
        r'\bgua\b': 'saya',
        r'\bsebesar\b': 'besar',
        r'\btangkaap\b': 'tangkap',
        r'\bdlm\b': 'dalam',
        r'\byakinn\b': 'yakin',
        r'\bslalu\b':'selalu',
        r'\bsdh\b': 'sudah',
        r'\btp\b': 'tapi',
        r'\bperedaranya\b': 'peredarannya',
        r'\bmembranntas\b': 'memberantas',
        r'\bbgd\b': 'sangat',
        r'\byg\b': 'yang',
        r'\budh\b': 'sudah',
        r'\bkalo\b':'kalau',
        r'\bjdi\b': 'jadi',
        r'\bdgn\b': 'dengan',
        r'\bipolisi\b': 'polisi',
        r'\bktemu\b':'ketemu',
        r'\bbahi\b':'bagi',
        r'\bdlam\b':'dalam',
        r'\bmngedsrkn\b':'dengar',
        r'\biknum\b':'oknum',
        r'\bbrang\b': 'barang',
        r'\btrlihat\b': 'lihat', 
        r'\bsbelum\b':'belum',
        r'\btrjadi\b':'jadi',
        r'\bkluar\b': 'keluar',
        r'\bbarbuk\b':'barang bukti',
        r'\bprtanyaan\b':'tanya',
        r'\basarakat\b': 'masyarakat',
        r'\btsb\b': 'tersebut',
        r'\bhukom\b': 'hukum',
        r'\bknp\b': 'kenapa',
        r'\babis\b': 'habis',
        r'\bmslah\b': 'masalah',
        r'\bkluarga\b': 'keluarga',
        r'\bklaurganya\b': 'keluarga',
        r'\bkplri\b': 'kapolri',
        r'\bbngt\b': 'sangat',
        r'\bkpn\b': 'kapan',
        r'\bexekusi\b': 'eksekusi',
        r'\bterta gkap\b': 'tertangkap',
        'lengksp':'lengkap',
        'kerennnn':'bagus',
        'nombor': 'nomor',
        'kerana':'karena',
        'ngedar':'edar',
        'kusus': 'khusus',
    }
    

    for key, value in normalization.items():
        words = [re.sub(key, value, word) for word in words]

    # Stemming
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    words = [stemmer.stem(word) for word in words]

    # Words post-processing
    words = [re.sub(r'nya$', '', word) if word.endswith('nya') else word for word in words]
    words = [re.sub(r'tt$', '', word) if word.endswith('tt') else word for word in words]
    words = [re.sub(r'kann$', '', word) if word.endswith('kann') else word for word in words]
    words = [re.sub(r'\d+$', '', word) if word.endswith(tuple('0123456789')) else word for word in words]
    words = [re.sub(r'\b\d+', '', word) for word in words]

    # Token Filtering
    filtered_words = []
    for word in words:
        if len(word) > 2:
            filtered_words.append(word)

    words = filtered_words

    # Return only if words count is greater than 10
    if len(words) <= 10:
        return None

    return ' '.join(words)