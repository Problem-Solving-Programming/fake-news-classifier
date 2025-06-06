import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

model_path = "./bert-fake-news-model"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

title = "As U.S. budget fight looms, Republicans flip their fiscal script"
text = """
WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a ?쐄iscal conservative??on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS???쏤ace the Nation,??drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense ?쐂iscretionary??spending on programs that support education, scientific research, infrastructure, public health and environmental protection. ?쏷he (Trump) administration has already been willing to say: ?쁗e?셱e going to increase non-defense discretionary spending ... by about 7 percent,?쇺?Meadows, chairman of the small but influential House Freedom Caucus, said on the program. ?쏯ow, Democrats are saying that?셲 not enough, we need to give the government a pay raise of 10 to 11 percent. For a fiscal conservative, I don?셳 see where the rationale is. ... Eventually you run out of other people?셲 money,??he said. Meadows was among Republicans who voted in late December for their party?셲 debt-financed tax overhaul, which is expected to balloon the federal budget deficit and add about $1.5 trillion over 10 years to the $20 trillion national debt. ?쏧t?셲 interesting to hear Mark talk about fiscal responsibility,??Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican tax bill would require the  United States to borrow $1.5 trillion, to be paid off by future generations, to finance tax cuts for corporations and the rich. ?쏷his is one of the least ... fiscally responsible bills we?셶e ever seen passed in the history of the House of Representatives. I think we?셱e going to be paying for this for many, many years to come,??Crowley said. Republicans insist the tax package, the biggest U.S. tax overhaul in more than 30 years,  will boost the economy and job growth. House Speaker Paul Ryan, who also supported the tax bill, recently went further than Meadows, making clear in a radio interview that welfare or ?쐃ntitlement reform,??as the party often calls it, would be a top Republican priority in 2018. In Republican parlance, ?쐃ntitlement??programs mean food stamps, housing assistance, Medicare and Medicaid health insurance for the elderly, poor and disabled, as well as other programs created by Washington to assist the needy. Democrats seized on Ryan?셲 early December remarks, saying they showed Republicans would try to pay for their tax overhaul by seeking spending cuts for social programs. But the goals of House Republicans may have to take a back seat to the Senate, where the votes of some Democrats will be needed to approve a budget and prevent a government shutdown. Democrats will use their leverage in the Senate, which Republicans narrowly control, to defend both discretionary non-defense programs and social spending, while tackling the issue of the ?쏡reamers,??people brought illegally to the country as children. Trump in September put a March 2018 expiration date on the Deferred Action for Childhood Arrivals, or DACA, program, which protects the young immigrants from deportation and provides them with work permits. The president has said in recent Twitter messages he wants funding for his proposed Mexican border wall and other immigration law changes in exchange for agreeing to help the Dreamers. Representative Debbie Dingell told CBS she did not favor linking that issue to other policy objectives, such as wall funding. ?쏻e need to do DACA clean,??she said.  On Wednesday, Trump aides will meet with congressional leaders to discuss those issues. That will be followed by a weekend of strategy sessions for Trump and Republican leaders on Jan. 6 and 7, the White House said. Trump was also scheduled to meet on Sunday with Florida Republican Governor Rick Scott, who wants more emergency aid. The House has passed an $81 billion aid package after hurricanes in Florida, Texas and Puerto Rico, and wildfires in California. The package far exceeded the $44 billion requested by the Trump administration. The Senate has not yet voted on the aid. 
"""

combined_input = title + " " + text
encoding = tokenizer(combined_input, truncation=True, padding=True, return_tensors="pt", max_length=512)

with torch.no_grad():
    outputs = model(**encoding)
    probs = F.softmax(outputs.logits, dim=1)

fake_prob = probs[0][0].item()
real_prob = probs[0][1].item()

print("🧠 예측 결과:", "FAKE" if fake_prob > real_prob else "REAL")
print(f"🔍 FAKE 확률: {fake_prob * 100:.2f}%")
print(f"🔍 REAL 확률: {real_prob * 100:.2f}%")
