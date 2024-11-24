# dataset modification/cleaning for bridge 
handsData <- read.csv("C:/Users/liapu/OneDrive/Desktop/Fall 2024/Artificial Intelligence/Wroclaw 2022 Championship Database CSV fixed/Hands.csv")
boardResults <- read.csv("C:/Users/liapu/OneDrive/Desktop/Fall 2024/Artificial Intelligence/Wroclaw 2022 Championship Database CSV fixed/BoardResults.csv")
pairHands <- read.csv("C:/Users/liapu/OneDrive/Desktop/Fall 2024/Artificial Intelligence/Wroclaw 2022 Championship Database CSV fixed/PairsHands.csv")
pairsResults <- read.csv("C:/Users/liapu/OneDrive/Desktop/Fall 2024/Artificial Intelligence/Wroclaw 2022 Championship Database CSV fixed/PairsResults.csv")

#ids in both groups
b_res_only <- anti_join(boardResults, handsData, 
                         by = c("boardcd" = "BrdCd",
                                "boardNumber" = "BrdNr"))

# Find rows in df2 not in df1
hands_only_0 <- anti_join(handsData, boardResults, 
                        by = c("BrdCd"="boardcd",
                               "BrdNr"="boardNumber"))
intersect_vector <- intersect(handsData$BrdCd, boardResults$boardcd)

#filtering out ids not in intersect
boardFilter <- boardResults |>
filter(boardcd %in% intersect_vector)

#double check
boardFilter <- boardResults |>
  filter(!(boardcd %in% b_res_only$boardcd & 
          boardNumber %in% b_res_only$boardNumber))

#seeing what's going on with these ids
lost_ids_boardRes <- setdiff(boardResults$boardcd, intersect_vector) 

print(lost_ids_boardRes)
#we do not have hand data for Finals I presume! (no ids in hands not in boards)

#Note that the tournament took the duplicate format, suggesting that each hand
#played multiple times

boardFilter_contracts <- boardFilter|>
  group_by(boardcd, boardNumber,ClosedContract) |>
  summarize(numCon = n(),
            avgTricks = round(mean(ClosedTricks)))

hands_count <- handsData |>
  group_by(BrdCd) |>
  summarize(numPlayed = n())

#most popular contract for each hand
boardFilter_max <- boardFilter_contracts |>
  group_by(boardcd, boardNumber) |>
  filter(numCon == max(numCon)) |>
  ungroup()



#left joining the dataSets
masterSet <- left_join(handsData, boardFilter_max, 
                       by = c("BrdCd"="boardcd",
                              "BrdNr"="boardNumber"))

#looking at the lost rows
extraRows <- boardFilter_max |>
  group_by(boardcd, boardNumber) |>
  summarize(numRowsExtra = n()) |>
  filter(numRowsExtra > 1)

boards_extra <- boardResults |>
  filter(boardcd %in% extraRows$boardcd &
        boardNumber %in% extraRows$boardNumber) |>
  arrange(boardcd, boardNumber)

dataSet <- masterSet[, 4:29]



# Function to calculate HCP from a given string of cards
calc_hcp <- function(cards) {
  #  table for point values
  points <- c('A' = 4, 'K' = 3, 'Q' = 2, 'J' = 1)
  
 
  hcp <- 0
  
  # char loop
  for (card in strsplit(cards, NULL)[[1]]) {
    # if in add pts
    if (card %in% names(points)) {
      hcp <- hcp + points[card]
    }
  }
  
  return(hcp)
}

#mutating on HCP
dataSet <- dataSet |>
  mutate(HCP_N = sapply(paste(NS, NH, ND, NC), calc_hcp),
         HCP_E = sapply(paste(ES, EH, ED, EC), calc_hcp),
         HCP_S = sapply(paste(SS, SH, SD, SC), calc_hcp),
         HCP_W = sapply(paste(WS, WH, WD, WC), calc_hcp))


#function to determine the trump suit
trump_suit <- function(contract) {
  for (str in c("S","H","D","C","NT")) {
    if(str_detect(contract, str))  {
      return(str)
    }
  }
  return(NULL)
}

dist_points <- function(contract, suits) {
  trump <- trump_suit(contract)
  dist <- 0
  counter <- 0
  suit_track <- c("S" =0, "H" =1, "D" =2, "C"=3)
  points <- c("0"=3 , "1"=2, "2"=1)
  if (trump == "NT") {
    return(0)
  }
  else {
    for (suit in suits) {
      if(counter == suit_track[trump]) {
        next
      }
      # Calculate the length of the suit (number of cards in the suit)
      suit_length <- length(strsplit(suit, NULL)[[1]])
      
      # Check if the suit length is in the points table
      if (as.character(suit_length) %in% names(points)) {
        dist <- dist + points[as.character(suit_length)]
      }
      counter <- counter + 1
    }
  }
  return(dist)
}

#mutating on distribution points
#292 lacks closedcontract

dataSet <- dataSet |>
  slice(-292) |>
  rowwise() |>
  mutate(Dist_N = dist_points(ClosedContract, c(NS, NH, ND, NC)),
         Dist_E = dist_points(ClosedContract, c(ES, EH, ED, EC)),
         Dist_S = dist_points(ClosedContract, c(SS, SH, SD, SC)),
         Dist_W = dist_points(ClosedContract, c(WS, WH, WD, WC))) |>
  ungroup()


## add totalPoints Column

dataSet <- dataSet |>
  mutate(TotalPoints_N = Dist_N + HCP_N,
         TotalPoints_E = Dist_E + HCP_E,
         TotalPoints_S = Dist_S + HCP_S,
         TotalPoints_W = Dist_W + HCP_W)

# add unprotected honors

unprotected_hands <- function(hand) {
  result <- character(0)
  for (suit in hand) {
    result <- c(result, unprotected_honors(suit))
  }
return(result)
}

unprotected_honors <- function(suit) {
  suit_length <- nchar(suit)
  unprotected_cards <- character(0)
  if(suit_length <= 3 & str_detect(suit, "J")) {
    unprotected_cards <- c("J")
  }
  if(suit_length <= 2 & str_detect(suit, "Q")) {
    unprotected_cards <- c(unprotected_cards, "Q")
  }
  if(suit_length <= 1 & str_detect(suit, "K")) {
    unprotected_cards <- c(unprotected_cards, "K")
  }
  if(all(c("Q", "K") %in% unprotected_cards) |
     all(c("J", "Q", "K") %in% unprotected_cards)) {
    unprotected_cards <- c()
  }
  if(str_detect(suit, "A")) {
    unprotected_cards <- c()
  }
  return(unprotected_cards)
}

#unprotected honors added to the set
dataSet <- dataSet |>
  rowwise() |>
  mutate(
    unprot_N = list(unprotected_hands(c(NS, NH, ND, NC))),
    unprot_E = list(unprotected_hands(c(ES, EH, ED, EC))),
    unprot_S = list(unprotected_hands(c(SS, SH, SD, SC))),
    unprot_W = list(unprotected_hands(c(WS, WH, WD, WC)))) |>
  ungroup()

#positioning (if the person that goes after you has more than ten points)

dataSet <- dataSet |>
  mutate(pos_N = (HCP_E >= 10),
         pos_E = (HCP_S >= 10),
         pos_S = (HCP_W >= 10),
         pos_W = (HCP_N >= 10))

# len trump function
len_trump <- function(hand, contract) {
  contract_suit <- trump_suit(contract)
  if(contract_suit == "NT")
    return(NA)
  suit_track <- c("S" =0, "H" =1, "D" =2, "C"=3)
  counter <- 0
  for(suit in hand) {
    if(counter == suit_track[contract_suit]) {
       return(str_length(suit))
    }
    counter <- counter + 1
  }
  return(0)
}
#Length of the trump suit
dataSet <- dataSet |>
  rowwise() |>
  mutate(Len_Trump_N = len_trump(c(NS, NH, ND, NC), ClosedContract),
         Len_Trump_E = len_trump(c(ES, EH, ED, EC), ClosedContract),
         Len_Trump_S = len_trump(c(SS, SH, SD, SC), ClosedContract),
         Len_Trump_W = len_trump(c(WS, WH, WD, WC), ClosedContract)) |>
  ungroup()

# ration of Aces points 

ratio_aces <- function(hand, hcp) {
  if(hcp ==0) {
    return(1)
  }
  numAces <- str_count(hand, "A")
  return(numAces * 4/hcp)
}

#adding to the dataSet

dataSet <- dataSet |>
  rowwise() |>
  mutate(
    Ace_Rat_N = ratio_aces(paste(NS, NH, ND, NC), HCP_N),
    Ace_Rat_E = ratio_aces(paste(ES, EH, ED, EC), HCP_E),
    Ace_Rat_S = ratio_aces(paste(SS, SH, SD, SC), HCP_S),
    Ace_Rat_W = ratio_aces(paste(WS, WH, WD, WC), HCP_W)) |>
  ungroup()
   
    
library(writexl)

# Save the dataset as an Excel file
write_xlsx(dataSet, "BridgeData2.xlsx")
