#packages

# dataset modification/cleaning for bridge 
handsData <- read.csv("C:/Users/liapu/OneDrive/Desktop/Fall 2024/Artificial Intelligence/Wroclaw 2022 Championship Database CSV fixed/Hands.csv")
boardResults <- read.csv("C:/Users/liapu/OneDrive/Desktop/Fall 2024/Artificial Intelligence/Wroclaw 2022 Championship Database CSV fixed/BoardResults.csv")


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
  mutate(N_HCP = sapply(paste(NS, NH, ND, NC), calc_hcp),
         E_HCP = sapply(paste(ES, EH, ED, EC), calc_hcp),
         S_HCP = sapply(paste(SS, SH, SD, SC), calc_hcp),
         W_HCP = sapply(paste(WS, WH, WD, WC), calc_hcp))


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
        counter <- counter + 1
        next
      }
      # Calculate the length of the suit (number of cards in the suit)
      suit_length <- length(strsplit(suit, NULL)[[1]])
      
      # Check if the suit length is in the points table
      if (as.character(suit_length) %in% names(points)) {
        dist <- dist + points[as.character(suit_length)]
      }
      
    }
  }
  return(dist)
}

#mutating on distribution points
#292 lacks closedcontract

dataSet <- dataSet |>
  slice(-292) |>
  rowwise() |>
  mutate(N_Dist = dist_points(ClosedContract, c(NS, NH, ND, NC)),
         E_Dist = dist_points(ClosedContract, c(ES, EH, ED, EC)),
         S_Dist = dist_points(ClosedContract, c(SS, SH, SD, SC)),
         W_Dist = dist_points(ClosedContract, c(WS, WH, WD, WC))) |>
  ungroup()


## add totalPoints Column

dataSet <- dataSet |>
  mutate(N_TotalPoints = N_Dist + N_HCP,
         E_TotalPoints = E_Dist + E_HCP,
         S_TotalPoints = S_Dist + S_HCP,
         W_TotalPoints = W_Dist + W_HCP)

# add unprotected honors

library(stringr)

unprotected_hands <- function(hand) {
  result <- ""
  for (suit in hand) {
    result <- paste0(result, unprotected_honors(suit))
  }
  return(result)
}

unprotected_honors <- function(suit) {
  unprotected_cards <- ""
  suit_length <- nchar(suit)
  
  # Check for unprotected honors, based on the suit length and honor card presence
  if (str_detect(suit, "A")) {
    return("")  # Ace is never unprotected
  }
  
  if (str_detect(suit, "K") && suit_length <= 1) {
    unprotected_cards <- paste0(unprotected_cards, "K")
  }
  
  if (str_detect(suit, "Q") && suit_length <= 2) {
    unprotected_cards <- paste0(unprotected_cards, "Q")
  }
  
  if (str_detect(suit, "J") && suit_length <= 3) {
    unprotected_cards <- paste0(unprotected_cards, "J")
  }
  
  # Additional logic to handle cases where cards are protected
  if (str_detect(suit, "K") && !str_detect(suit, "Q")) {
    return(unprotected_cards)  # Only return unprotected cards when no Queen
  }
  if (str_detect(suit, "Q") && !str_detect(suit, "J")) {
    return(unprotected_cards)  # Only return unprotected cards when no Jack
  }
  
  return(unprotected_cards)
}


#unprotected honors added to the set
dataSet <- dataSet |>
  rowwise() |>
  mutate(
    N_unprot = unprotected_hands(c(NS, NH, ND, NC)),
    E_unprot = unprotected_hands(c(ES, EH, ED, EC)),
    S_unprot = unprotected_hands(c(SS, SH, SD, SC)),
    W_unprot = unprotected_hands(c(WS, WH, WD, WC))) |>
  ungroup()

#positioning (if the person that goes after you has more than ten points)

dataSet <- dataSet |>
  mutate(N_pos = (E_HCP >= 10),
         E_pos = (S_HCP >= 10),
         S_pos = (W_HCP >= 10),
         W_pos = (N_HCP >= 10))

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
  mutate(N_Len_Trump = len_trump(c(NS, NH, ND, NC), ClosedContract),
         E_Len_Trump = len_trump(c(ES, EH, ED, EC), ClosedContract),
         S_Len_Trump = len_trump(c(SS, SH, SD, SC), ClosedContract),
         W_Len_Trump = len_trump(c(WS, WH, WD, WC), ClosedContract)) |>
  ungroup()

# ration of Aces points 

ratio_aces <- function(hand, hcp) {
  if(hcp ==0) {
    return(1)
  }
  numAces <- str_count(hand, "A")
  return(numAces * 4/hcp)
}

#adding to the dataSet ace ratio

dataSet <- dataSet |>
  rowwise() |>
  mutate(
    N_Ace_Rat = ratio_aces(paste(NS, NH, ND, NC), N_HCP),
    E_Ace_Rat = ratio_aces(paste(ES, EH, ED, EC), E_HCP),
    S_Ace_Rat = ratio_aces(paste(SS, SH, SD, SC), S_HCP),
    W_Ace_Rat = ratio_aces(paste(WS, WH, WD, WC), W_HCP)) |>
  ungroup()


#adding the trump suit to the dataSet

dataSet <- dataSet |>
  rowwise() |>
  mutate(trump = trump_suit(ClosedContract)) |>
  ungroup()

#adding the diff between #tricks and contract
dataSet <- dataSet |>
  filter(!ClosedContract == "PASS") |>
  mutate(diff_tricks_contract =
           avgTricks - (as.numeric(substr(ClosedContract, 1, 1)) + 6)) 

#filteringthe Data Sets into E/W Declarer and N/S Declarer --> making all declarers N/S

EW_dec_set <- dataSet |>
  filter(Dlr == "E" |Dlr == "W")

NS_dec_set <- dataSet |>
  filter(Dlr == "N" |Dlr == "S")


# Assuming EW_dec_set is your data frame
columns <- colnames(EW_dec_set)
# Replace the prefixes in the column names
for (i in 1:length(columns)) {
  if (substr(columns[i], 1, 1) == "N") {
    print(columns[i])
    columns[i]<- sub("^N", "S", columns[i])
    print(columns[i])
  }
  else if (substr(columns[i], 1, 1) == "S") {
    print(columns[i])
    columns[i]<- sub("^S", "N", columns[i])
    print(columns[i])
  }
  else if (substr(columns[i], 1, 1) == "E") {
    print(columns[i])
    columns[i]<- sub("^E", "W", columns[i])
    print(columns[i])
  }
  else if (substr(columns[i], 1, 1) == "W") {
    print(columns[i])
    columns[i]<- sub("^W", "E", columns[i])
    print(columns[i])
  }
}

# Assign the modified column names back to the data frame
colnames(EW_dec_set) <- columns

finalSet <- rbind(EW_dec_set, NS_dec_set)

finalSet <- finalSet |>
  select(-c(SS,SH,SD,SC,NS,NH,ND,NC,ES,EH,ED,EC,WS,WH,WD,WC,
            tourndetailaa, Round, BrdNr, BrdCd, Phase, numCon,
            Dlr, Vuln, ClosedContract, diff_tricks_contract,
            S_Dist, N_Dist, E_Dist, W_Dist))

#discretizing TotalPoints (HCP + Dist points)

finalSet <- finalSet |>
  mutate(across(ends_with("TotalPoints"), ~ case_when(
    . <= 6 ~ 0,
    . <= 12 ~ 1,
    . <= 17 ~ 2,
    TRUE ~ 3)))

#discretizing HCP
finalSet <- finalSet |>
  mutate(across(ends_with("HCP"), ~ case_when(
    . <= 6 ~ 0,
    . <= 12 ~ 1,
    . <= 17 ~ 2,
    TRUE ~ 3)))
  #discretizing ACE_ratio
finalSet <- finalSet |> 
  mutate(across(ends_with("Ace_Rat"), ~ case_when(
    . <= .75 ~ 0,
    TRUE ~ 1)))

#Separate into NT and Trump contract data Sets 

NT_data <- finalSet %>%
  filter(trump == "NT") %>%
  select(-c(N_TotalPoints, W_TotalPoints, S_TotalPoints, E_TotalPoints, 
            N_Len_Trump, S_Len_Trump, W_Len_Trump, E_Len_Trump, trump))

Trump_data <- finalSet %>%
  filter(trump != "NT") %>%
  select(-c(N_HCP, W_HCP, S_HCP, E_HCP)) 

# Save the dataset as an Excel file
write_csv(NT_data, "BridgeDataTrump.csv")
write_csv(Trump_data, "BridgeDataNoTrump.csv")



