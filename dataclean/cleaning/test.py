from simhash import Simhash

pattern_punctuation = r"""[!?,.:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""



s1=Simhash('இதனால் கடுமையான எரிபொருள் பற்றாக்குறை நிலவுகிறது.', reg=r'[\w]+|{}'.format(pattern_punctuation))
s2=Simhash('இலங்கையின் நாணயம் பெரிய அளவில் மதிப்பிழந்து விட்டது.', reg=r'[\w]+|{}'.format(pattern_punctuation))


print(s1.value)
print(s2.value)
print(s1.distance(s2))
